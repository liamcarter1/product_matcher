"""Tests for tools/vision_extract.py — image-based model code extraction."""

import pytest
from unittest.mock import patch, MagicMock

from tools.vision_extract import extract_text_from_image, VISION_MODEL


class TestExtractTextFromImage:

    def test_file_not_found(self):
        result = extract_text_from_image("/nonexistent/path.jpg")
        assert result["error"] is not None
        assert "not found" in result["error"].lower()
        assert result["text"] == ""

    def test_unsupported_file_type(self, tmp_path):
        bmp_file = tmp_path / "test.bmp"
        bmp_file.write_bytes(b"BM" + b"\x00" * 100)
        result = extract_text_from_image(str(bmp_file))
        assert result["error"] is not None
        assert "Unsupported" in result["error"]

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        result = extract_text_from_image(str(empty))
        assert result["error"] is not None
        assert "empty" in result["error"].lower()

    def test_oversized_file(self, tmp_path):
        big = tmp_path / "big.jpg"
        big.write_bytes(b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024))
        result = extract_text_from_image(str(big))
        assert result["error"] is not None
        assert "too large" in result["error"].lower()

    @patch("tools.vision_extract._get_client")
    def test_successful_extraction(self, mock_get_client, tmp_path):
        img = tmp_path / "label.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "4WE6D6X/EG24N9K4\nBosch Rexroth"
        mock_get_client().chat.completions.create.return_value = mock_response

        result = extract_text_from_image(str(img))

        assert result["error"] is None
        assert "4WE6D6X/EG24N9K4" in result["text"]
        assert "Bosch Rexroth" in result["text"]
        assert result["raw_response"] == "4WE6D6X/EG24N9K4\nBosch Rexroth"

    @patch("tools.vision_extract._get_client")
    def test_model_code_only(self, mock_get_client, tmp_path):
        img = tmp_path / "label.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "D1VW020BN"
        mock_get_client().chat.completions.create.return_value = mock_response

        result = extract_text_from_image(str(img))

        assert result["error"] is None
        assert result["text"] == "D1VW020BN"

    @patch("tools.vision_extract._get_client")
    def test_unreadable_image(self, mock_get_client, tmp_path):
        img = tmp_path / "blur.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "UNREADABLE"
        mock_get_client().chat.completions.create.return_value = mock_response

        result = extract_text_from_image(str(img))

        assert result["error"] is not None
        assert "clearer photo" in result["error"].lower()
        assert result["text"] == ""

    @patch("tools.vision_extract._get_client")
    def test_api_failure_handled_gracefully(self, mock_get_client, tmp_path):
        img = tmp_path / "label.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_get_client().chat.completions.create.side_effect = Exception("API down")

        result = extract_text_from_image(str(img))

        assert result["error"] is not None
        assert "try again" in result["error"].lower()

    @patch("tools.vision_extract._get_client")
    def test_uses_correct_model_and_detail(self, mock_get_client, tmp_path):
        img = tmp_path / "label.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "TEST-123"
        mock_client = mock_get_client()
        mock_client.chat.completions.create.return_value = mock_response

        extract_text_from_image(str(img))

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == VISION_MODEL
        messages = call_kwargs.kwargs["messages"]
        image_content = [c for c in messages[0]["content"] if c.get("type") == "image_url"]
        assert len(image_content) == 1
        assert image_content[0]["image_url"]["detail"] == "high"

    @patch("tools.vision_extract._get_client")
    def test_specs_in_response_not_in_query(self, mock_get_client, tmp_path):
        """Specs lines (with ':') should NOT be included in the query text."""
        img = tmp_path / "label.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "4WE6D6X/EG24N9K4\nBosch Rexroth\nvoltage: 24VDC\npressure: 315 bar"
        )
        mock_get_client().chat.completions.create.return_value = mock_response

        result = extract_text_from_image(str(img))

        assert result["error"] is None
        assert result["text"] == "4WE6D6X/EG24N9K4 Bosch Rexroth"
        assert "voltage" not in result["text"]
