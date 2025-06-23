import random
import openai
import json

from langchain.docstore.document import Document as LangChainDocument
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from fastapi import HTTPException
from uuid import UUID, uuid4
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from sqlmodel import Session, text
from util import sanitize_input, sanitize_output
from langchain import OpenAI
from typing import List, Union, Optional, Dict, Tuple, Any
from helpers import (
    get_user_by_uuid_or_identifier,
    get_chat_session_by_uuid,
)

from constants import COMPONENTS

from models import (
    User,
    Organization,
    Project,
    Node,
    ChatSession,
    ChatSessionResponse,
    TrimmedConversationBufferMemory,
    get_engine,
)
from config import (
    CHANNEL_TYPE,
    DOCUMENT_TYPE,
    ENABLE_CACHE_ANSWER,
    LLM_MODELS,
    LLM_DISTANCE_THRESHOLD,
    LLM_DEFAULT_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_CHUNK_SIZE,
    LLM_CHUNK_OVERLAP,
    LLM_MIN_NODE_LIMIT,
    LLM_DEFAULT_DISTANCE_STRATEGY,
    VECTOR_EMBEDDINGS_COUNT,
    DISTANCE_STRATEGY,
    AGENT_NAMES,
    API_KEY,
    SU_DSN,
    logger,
)

chat_history: Dict[str, TrimmedConversationBufferMemory] = {}
chat_history_summary: Dict[str, str] = {}

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings(openai_api_key=API_KEY)

# Create the vector store (table will be auto-created)
get_engine(dsn=SU_DSN)

collection_name = "answered_questions"
vectorstore = PGVector(
    collection_name=collection_name,
    connection_string=SU_DSN,
    embedding_function=embedding,
)

procedures = [
    "khai sinh",
    "thường trú",
    "kết hôn",
    "khai tử",
    "cấp lại cccd",
    "cấp cccd",
    "cấp căn cước",
    "cấp lại căn cước",
    "giám hộ",
    "nhận con nuôi",
    "đăng ký",
    "thủ tục",
    "hướng dẫn",
    "cơ quan",
    "cổng dịch vụ công",
]


# -------------
# Query the LLM
# -------------
def chat_query(
    query_str: str,
    session_id: Optional[Union[str, UUID]] = None,
    meta: Optional[Dict[str, Any]] = {},
    channel: Optional[CHANNEL_TYPE] = None,
    identifier: Optional[str] = None,
    project: Optional[Project] = None,
    organization: Optional[Organization] = None,
    session: Optional[Session] = None,
    user_data: Optional[Dict[str, Any]] = None,
    distance_strategy: Optional[DISTANCE_STRATEGY] = DISTANCE_STRATEGY.EUCLIDEAN,
    distance_threshold: Optional[float] = LLM_DISTANCE_THRESHOLD,
    node_limit: Optional[int] = LLM_MIN_NODE_LIMIT,
    model: Optional[LLM_MODELS] = LLM_MODELS.GPT_35_TURBO,
) -> ChatSessionResponse:
    """
    Steps:
        1. ✅ Clean user input
        2. ✅ Get chat summary
        3. ✅ Check for cached answer
        4. ✅ If the answer is found, proceed to step 10
        5. ✅ Use router to determine the strategy
        6. ✅ Create input embeddings
        7. ✅ Search for similar nodes
        8. ✅ Create prompt template w/ similar nodes
        9. ✅ Submit prompt template to LLM
        10. ✅ Get response from LLM
        11. ✅ Store response in vector store
        12. Create ChatSession
            - Store embeddings
            - Store is_escalate
        13. Return response
    """
    if query_str is None:
        raise ValueError("query_str is required")
        return None

    meta = {}
    agent_name = None
    embeddings = []
    is_escalate = False
    response_message = None
    prompt = None
    context_str = None
    query_embeddings = None
    cached_metadata = None
    images_list: List[str] = []
    MODEL_TOKEN_LIMIT = (
        model.token_limit if isinstance(model, OpenAI) else LLM_MAX_OUTPUT_TOKENS
    )

    # ---------------------------------------------
    # Generate a new session ID if none is provided
    # ---------------------------------------------
    prev_chat_session = (
        get_chat_session_by_uuid(session_id=session_id, session=session)
        if session_id
        else None
    )

    # If we were given an invalid session_id
    if session_id and not prev_chat_session:
        return HTTPException(
            status_code=404, detail=f"Chat session with ID {session_id} not found."
        )
    # If we were given a valid session_id
    elif session_id and prev_chat_session and prev_chat_session.meta.get("agent"):
        agent_name = prev_chat_session.meta["agent"]
    # If this is a new session, generate a new ID
    else:
        session_id = str(uuid4())

    meta["agent"] = agent_name if agent_name else random.choice(AGENT_NAMES)

    # ----------------
    # Get user details
    # ----------------
    user = get_user_by_uuid_or_identifier(
        identifier, session=session, should_except=False
    )

    if not user:
        logger.debug("🚫👤 User not found, creating new user")
        user_params = {
            "identifier": identifier,
            "identifier_type": channel.value
            if isinstance(channel, CHANNEL_TYPE)
            else channel,
        }
        if user_data:
            user_params = {**user_params, **user_data}

        user = User.create(user_params)
    else:
        logger.debug(f"👤 User found: {user}")

    if user.identifier not in chat_history:
        chat_history[user.identifier] = TrimmedConversationBufferMemory(
            memory_key="history"
        )

    if user.identifier not in chat_history_summary:
        chat_history_summary[user.identifier] = ""

    # ----------------
    # Clean user input
    # ----------------
    query_str = sanitize_input(query_str)
    logger.debug(f"💬 Query received: {query_str}")
    user_message = f"{query_str}"

    # ----------------------
    # Get chat summary
    # ----------------------
    if query_str:
        chat_summary = sanitize_input(
            retrieve_chat_summary(
                user_id=user.identifier,
                query_str=query_str,
                model=model,
            )
        )
        chat_history_summary[user.identifier] = chat_summary

    query_str = chat_summary  # + query_str
    logger.debug(f"💬 Query with summary: {query_str}")

    # ----------------
    # Check for cached answer
    # ----------------
    if ENABLE_CACHE_ANSWER:
        cached_metadata = get_cached_answer(query_str)
        logger.debug(f"💬 Cached answer: {cached_metadata}")

    if cached_metadata:
        response_message = cached_metadata["answer"]
        images_list = cached_metadata["images_list"]
    else:
        strategy = query_router(query_str)
        logger.debug(f"💬 Query strategy: {strategy}")

        if strategy == "None":
            # return default response
            response_message = "Hãy hỏi tôi về hành chính công, tôi sẽ giúp bạn tìm kiếm thông tin cần thiết."
            is_escalate = True
        else:
            # ----------------
            # Get token counts
            # ----------------
            query_token_count = get_token_count(query_str)
            prompt_token_count = 0

            # -----------------------
            # Create input embeddings
            # -----------------------
            _, embeddings = get_embeddings(query_str)

            query_embeddings = embeddings[0]

            # ------------------------
            # Search for similar nodes
            # ------------------------
            nodes = get_nodes_by_embedding(
                query_embeddings,
                node_limit,
                distance_strategy=distance_strategy
                if isinstance(distance_strategy, DISTANCE_STRATEGY)
                else LLM_DEFAULT_DISTANCE_STRATEGY,
                distance_threshold=distance_threshold,
                session=session,
            )

            if len(nodes) > 0:
                if (not project or not organization) and session:
                    # get document from Node via session object:
                    document = session.get(Node, nodes[0].id).document
                    project = document.project
                    organization = project.organization

                # ----------------------
                # Create prompt template
                # ----------------------

                # concatenate all nodes into a single string
                context_str = "\n\n".join([node.text for node in nodes])

                # -------------------------------------------
                # Let's make sure we don't exceed token limit
                # -------------------------------------------
                context_token_count = get_token_count(context_str)

                # ----------------------------------------------
                # if token count exceeds limit, truncate context
                # ----------------------------------------------
                if (
                    context_token_count + query_token_count + prompt_token_count
                ) > MODEL_TOKEN_LIMIT:
                    logger.debug("🚧 Exceeded token limit, truncating context")
                    token_delta = MODEL_TOKEN_LIMIT - (
                        query_token_count + prompt_token_count
                    )
                    context_str = context_str[:token_delta]

                # create prompt template
                system_prompt, user_prompt = get_prompt_template(
                    user_query=query_str,
                    context_str=context_str,
                    project=project,
                    organization=organization,
                    agent=agent_name,
                )

                prompt_token_count = get_token_count(prompt)
                token_count = (
                    context_token_count + query_token_count + prompt_token_count
                )

                # ---------------------------
                # Get response from LLM model
                # ---------------------------
                # It should return a JSON dict
                llm_response = json.loads(
                    retrieve_llm_response(
                        user_prompt,
                        model=model,
                        prefix_messages=system_prompt,
                    )
                )
                is_escalate = llm_response.get("is_escalate", False)
                images_list = llm_response.get("images_list", [])
                response_message = llm_response.get("message", None)

                if ENABLE_CACHE_ANSWER:
                    store_answered_question(
                        question=query_str,
                        answer=response_message,
                        images_list=images_list,
                    )
                    logger.debug(f"LLM response: {str(response_message)}")
            else:
                logger.info("🚫📝 No similar nodes found, returning default response")
                response_message = "Xin lỗi, tôi không thể hỗ trợ bạn."
                is_escalate = True

    chat_history[user.identifier].chat_memory.add_user_message(user_message)
    chat_history[user.identifier].chat_memory.add_ai_message(response_message)

    # -----------------------------------
    # Calculate input and response tokens
    # -----------------------------------
    token_count = get_token_count(prompt) + get_token_count(response_message)

    # ---------------
    # Add to meta tag
    # ---------------

    meta["is_escalate"] = is_escalate
    meta["images_list"] = images_list

    if session_id:
        meta["session_id"] = session_id

    chat_session = ChatSession(
        user_id=user.id,
        session_id=session_id,
        project_id=project.id if project else None,
        channel=channel.value if isinstance(channel, CHANNEL_TYPE) else channel,
        user_message=user_message,
        embeddings=query_embeddings,
        token_count=token_count if token_count > 0 else None,
        response=response_message,
        meta=meta,
    )

    if session:
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

    else:
        with Session(get_engine()) as session:
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)

    return chat_session


# ------------------------------
# Retrieve a random agent's name
# ------------------------------
def get_random_agent():
    return random.choice(AGENT_NAMES)


# ------------------------
# Retrieve prompt template
# ------------------------
def get_prompt_template(
    user_query: str = None,
    context_str: str = None,
    project: Optional[Project] = None,
    organization: Optional[Organization] = None,
    agent: str = None,
) -> str:
    agent = f"{agent}, " if agent else ""
    user_query = user_query if user_query else ""
    context_str = context_str if context_str else ""
    organization = (
        project.organization.display_name
        if project
        else organization.display_name
        if organization
        else None
    )

    if not context_str or not user_query:
        raise ValueError(
            "Missing required arguments context_str, user_query, organization, agent"
        )

    system_prompt = [
        {
            "role": "system",
            "content": f"""[AGENT]:
I am {agent} a very kind and enthusiastic customer support agent who loves to help customers. I am working on the behalf of "{organization}"

Given the following document from "{organization}", I will answer the [USER] questions using only the [DOCUMENT] and following the [RULES].

[DOCUMENT]:
{context_str}

[RULES]:
I will answer the user's questions using only the [DOCUMENT] provided. I will abide by the following rules:
- I am a kind and helpful human, the best customer support agent in existence
- I never lie or invent answers not explicitly provided in [DOCUMENT]
- If I am unsure of the answer response or the answer is not explicitly contained in [DOCUMENT], I will say: "Xin lỗi, tôi không thể hỗ trợ bạn trong lĩnh vực này.".
- I will always respond in JSON format with the following keys:
  + "message" my full, clear, concise answer in Vietnamese based only on the [DOCUMENT]; it must not contain image links/paths.
  + "is_escalate" a boolean, returning false if I am unsure and true if I do have a relevant answer
  + "images_list" a list of image URLs or file paths from the [DOCUMENT] that are relevant to the users question; they serve as visual aids but never as replacements for "message".
- I will only answer in Vietnamese
""",
        }
    ]

    return (system_prompt, f"[USER]:\n{user_query}")


# ----------------------------
# Get the count of tokens used
# ----------------------------
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def get_token_count(text: str):
    if not text:
        return 0

    return OpenAI().get_num_tokens(text=text)


# ------------------------
# Get the cached answer
# ------------------------
def get_cached_answer(query, distance_threshold=0.025) -> Optional[str]:
    """
    Check if the query is already in the vector store and return the cached answer if found.
    """
    results = vectorstore.similarity_search_with_score(query, k=1)
    if results:
        doc, score = results[0]
        # Lower score = more similar
        if score < distance_threshold:
            logger.debug(f"🔍 Found cached answer with score {score}")
            return doc.metadata
    return None


# ------------------------
# Store the answered question
# ------------------------
def store_answered_question(question: str, answer: str, images_list: list[str]):
    doc = Document(
        page_content=question,
        metadata={"answer": answer, "images_list": images_list},
    )
    vectorstore.add_documents([doc])


# --------------------------------------------
# Query embedding search for similar documents
# --------------------------------------------
def get_nodes_by_embedding(
    embeddings: List[float],
    k: int = LLM_MIN_NODE_LIMIT,
    distance_strategy: Optional[DISTANCE_STRATEGY] = LLM_DEFAULT_DISTANCE_STRATEGY,
    distance_threshold: Optional[float] = LLM_DISTANCE_THRESHOLD,
    session: Optional[Session] = None,
) -> List[Node]:
    # Convert embeddings array into sql string
    embeddings_str = str(embeddings)

    if distance_strategy == DISTANCE_STRATEGY.EUCLIDEAN:
        distance_fn = "match_node_euclidean"
    elif distance_strategy == DISTANCE_STRATEGY.COSINE:
        distance_fn = "match_node_cosine"
    elif distance_strategy == DISTANCE_STRATEGY.MAX_INNER_PRODUCT:
        distance_fn = "match_node_max_inner_product"
    else:
        raise Exception(f"Invalid distance strategy {distance_strategy}")

    # ---------------------------
    # Lets do a similarity search
    # ---------------------------
    sql = f"""SELECT * FROM {distance_fn}(
    '{embeddings_str}'::vector({VECTOR_EMBEDDINGS_COUNT}),
    {float(distance_threshold)}::double precision,
    {int(k)});"""

    # logger.debug(f'🔍 Query: {sql}')

    # Execute query, convert results to Node objects
    if not session:
        with Session(get_engine()) as session:
            nodes = session.exec(text(sql)).all()
    else:
        nodes = session.exec(text(sql)).all()

    return [Node.by_uuid(str(node[0])) for node in nodes] if nodes else []


# --------------
# Queries OpenAI
# --------------
def retrieve_llm_response(
    query_str: str,
    model: Optional[LLM_MODELS] = LLM_MODELS.GPT_35_TURBO,
    temperature: Optional[float] = LLM_DEFAULT_TEMPERATURE,
    max_output_tokens: Optional[int] = LLM_MAX_OUTPUT_TOKENS,
    prefix_messages: Optional[List[dict]] = None,
    ## user: Optional[User] = None,
):
    llm = OpenAI(
        temperature=temperature,
        model_name=model.model_name
        if isinstance(model, LLM_MODELS)
        else LLM_MODELS.GPT_35_TURBO.model_name,
        max_tokens=max_output_tokens,
        prefix_messages=prefix_messages,
    )
    try:
        result = llm(prompt=query_str)
        ## Conversation = ConversationChain(
        ##     llm=llm,
        ##     memory=chat_history[user.identifier],
        ## )

        ## result = Conversation.run(input=query_str)
    except openai.error.InvalidRequestError as e:
        logger.error(f"🚨 LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    logger.debug(f"💬 LLM result: {str(result)}")
    return sanitize_output(result)


# --------------------------
# Create document embeddings
# --------------------------
def get_embeddings(
    document_data: str,
    document_type: DOCUMENT_TYPE = DOCUMENT_TYPE.PLAINTEXT,
) -> Tuple[List[str], List[float]]:
    documents = [LangChainDocument(page_content=document_data)]

    logger.debug(documents)
    if document_type == DOCUMENT_TYPE.MARKDOWN:
        doc_splitter = MarkdownTextSplitter(
            chunk_size=LLM_CHUNK_SIZE, chunk_overlap=LLM_CHUNK_OVERLAP
        )
    else:
        doc_splitter = CharacterTextSplitter(
            chunk_size=LLM_CHUNK_SIZE, chunk_overlap=LLM_CHUNK_OVERLAP
        )

    # Returns an array of Documents
    split_documents = doc_splitter.split_documents(documents)

    # Lets convert them into an array of strings for OpenAI
    arr_documents = [doc.page_content for doc in split_documents]

    # https://github.com/hwchase17/langchain/blob/d18b0caf0e00414e066c9903c8df72bb5bcf9998/langchain/embeddings/openai.py#L219
    embed_func = OpenAIEmbeddings()

    embeddings = embed_func.embed_documents(
        texts=arr_documents, chunk_size=LLM_CHUNK_SIZE
    )

    return arr_documents, embeddings


def retrieve_chat_summary(
    user_id: Optional[str] = None,
    query_str: Optional[str] = None,
    model: Optional[LLM_MODELS] = LLM_MODELS.GPT_35_TURBO,
):
    previous_summary: str = chat_history_summary[user_id]
    previous_summary = previous_summary if previous_summary else "No previous summary"
    system_prompt = [
        {
            "role": "system",
            "content": f""":
You are updating a summary of a conversation. Using the previous summary as a background context. Follow these instructions:
- Always answer in Vietnamese.
- Rewrite the summary in a simple sentence so that it accurately reflects the conversation after the new message.
- If the current message shifts the topic completely, discard the old summary and generate a new one.

Previous summary:
"{previous_summary}"

Current user message:
"{query_str}"
""",
        }
    ]

    llm = OpenAI(
        temperature=0,
        model_name=model.model_name
        if isinstance(model, LLM_MODELS)
        else LLM_MODELS.GPT_35_TURBO.model_name,
        prefix_messages=system_prompt,
    )

    summary_prompt = "Update the summary to capture users intent."
    summary = llm(prompt=summary_prompt)

    return summary


def query_router(
    query_str: str,
):
    strategy = "None"

    lower_query = query_str.lower()

    for procedure in procedures:
        if procedure in lower_query:
            strategy = "RAG"
            return strategy

    return strategy
