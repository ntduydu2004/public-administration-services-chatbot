import json
import pytesseract

from langchain import OpenAI
from typing import List, Optional
from PIL import Image, ImageFile, ImageDraw, ImageFont
from config import LLM_MODELS, OUTPUT_HIGHLIGHT_PICTURE, USER_INPUT_PICTURE, logger

DISTANCE_X = 10
DISTANCE_Y = 20

class TextObject:
    text: str
    x: int
    y: int
    width: int
    height: int
    def __init__(self, text: str, x: int, y: int, width: int, height: int):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"TextObject(text={self.text}, x={self.x}, y={self.y}, width={self.width}, height={self.height})"

class HightLightQueryResponse:
    def __init__(self, text_object: TextObject, instruction: str):
        self.text_object = text_object
        self.instruction = instruction
    
    def __repr__(self):
        return f"HightLightQueryResponse(textObject={self.text_object}, instruction={self.instruction})"
        
def highlight_query(query: str, model: Optional[LLM_MODELS] = LLM_MODELS.GPT_35_TURBO):
    image = Image.open(USER_INPUT_PICTURE)
    
    texts = get_texts_from_ocr(image)
    
    response: HightLightQueryResponse = get_instruction_from_llm(texts, query)
    
    obj = response.text_object
    instructions = response.instruction
    
    converted_image = image.convert("RGB")
    highlight_boxes_with_instruction(converted_image, obj, instructions)
    
    
    
def highlight_boxes_with_instruction(image: Image.Image, obj: TextObject, instructions: str):
    x, y, w, h = obj.x, obj.y, obj.width, obj.height
    
    # Draw rectangle
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = x, y + h + 6, x + 350, y + h + 40

    # Load font (must support Vietnamese)
    font = ImageFont.truetype("CaskaydiaCoveNerdFont-Regular.ttf", 14)

    # Draw text label
    draw.text((x1 + 3, y1 + 2), instructions, font=font, fill="red")

    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    image.save(OUTPUT_HIGHLIGHT_PICTURE)

    

def get_texts_from_ocr(image: ImageFile):

    # Run Tesseract OCR with bounding box info
    ocr_data = pytesseract.image_to_data(image, lang='vie', output_type=pytesseract.Output.DICT)

    # Loop over each word and get its bounding box
    n_boxes = len(ocr_data['text'])
    
    results: List[TextObject] = []
    
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 0:  # Filter out low-confidence or empty results
            text = ocr_data['text'][i].strip()
            if text:
                x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                textObject = TextObject(text, x, y, w, h)
                results.append(textObject)
    
    return results

def get_instruction_from_llm(texts: List[TextObject], query: str, model: Optional[LLM_MODELS] = LLM_MODELS.GPT_35_TURBO,):
        system_prompt = [
            {
                "role": "system",
                "content": f"""
    You are a helpful assistant that highlights the most relevant text in an image based on user queries.
    Your task is to analyze the provided list of detected texts and the user's query, then return
    the one in that list which is the most relevant to the query. Here is the json format of it:
        {{
            "textObject": {{
                "text": "The text to highlight",
                "x": 100,
                "y": 150,
                "width": 200,
                "height": 50
            }},
            "instruction": "example of the necessary information"
        }}
    Examples about instruction:
        Example 1:
            user query: Hãy chỉ cho tôi vùng để điển thông tin về họ và tên
            instruction: Điền họ và tên, ví dụ Nguyễn Văn An
            
        Example 2:
            user query: Hãy chỉ cho tôi vùng để điền thông tin về số điện thoại
            instruction: Điền số điện thoại, ví dụ 0123456789
    
    If you cannot find any relevant text, return an empty object.
    You should return the instruction in Vietnamese.
    
    List of detected texts:
    {', '.join([f'"{text}"' for text in texts])}
    
    Current user query: "{query}"
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
    
        user_prompt = "Get the most relevant text object from the list above and return it in the specified format."
        response = json.loads(llm(prompt=user_prompt))
        
        json_text_object = response.get("textObject", {})
        instruction = response.get("instruction", "")
        
        text_object = TextObject(
            text=json_text_object.get("text", ""),
            x=json_text_object.get("x", None),
            y=json_text_object.get("y", None),
            width=json_text_object.get("width", None),
            height=json_text_object.get("height", None),
        )
        logger.debug(f"highlight query response: {response}")
        return HightLightQueryResponse(text_object, instruction)