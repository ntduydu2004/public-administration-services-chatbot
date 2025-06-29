import cv2
from config import OUTPUT_HIGHLIGHT_PICTURE, USER_INPUT_PICTURE
from doclayout_yolo import YOLOv10

model = YOLOv10("./models/ui-model/doclayout_yolo_docstructbench_imgsz1024.pt")


def highlight_input_boxes():
    # Perform prediction
    det_res = model.predict(
        USER_INPUT_PICTURE,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.05,  # Confidence threshold
        device="cpu",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    # Annotate and save the result
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite(OUTPUT_HIGHLIGHT_PICTURE, annotated_frame)
