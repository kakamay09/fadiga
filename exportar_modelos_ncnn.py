from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELO_PT = BASE_DIR / "train3" / "weights" / "best.pt"
IMG_SIZE = 224


def main() -> None:
    if not MODELO_PT.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODELO_PT}")

    modelo = YOLO(str(MODELO_PT))
    print(f"Modelo carregado: task={modelo.task}, classes={modelo.names}")

    print("Exportando ONNX...")
    onnx = modelo.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,
        simplify=True,
        dynamic=False,
    )
    print(f"ONNX gerado em: {onnx}")

    print("Exportando NCNN...")
    ncnn = modelo.export(format="ncnn", imgsz=IMG_SIZE)
    print(f"NCNN gerado em: {ncnn}")


if __name__ == "__main__":
    main()
