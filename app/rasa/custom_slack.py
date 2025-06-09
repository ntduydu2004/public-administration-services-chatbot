from rasa.core.channels.slack import SlackInput
from rasa.shared.utils import common as rasa_common
from typing import Any, Dict, Optional, Text
from sanic.request import Request

"""
The purpose of this script is to extend SlackInput to allow for custom metadata to be passed to Rasa.
"""


class CustomSlackInput(SlackInput):
    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        # For whatever reason, Rasa is unable to pass data via 'metadata' so 'meta' works for now
        metadata = request.json.get("message", {}).get("meta")

        # Debug
        rasa_common.logger.debug(f"[🤖 ActionGPTFallback]\nmetadata: {metadata}")
        return metadata if metadata is not None else None
