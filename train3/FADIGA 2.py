import os
import cv2
import time
import winsound

os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")
import vlc

from ultralytics import YOLO

# =========================
# CAMINHOS
# =========================
CAMINHO_MODELO = r"C:\Users\cmay\PycharmProjects\PythonProject\runs\classify\train3\weights\best.pt"
CAMINHO_FADIGA = r"C:\Users\cmay\PycharmProjects\PythonProject\fadiga.mp3"
CAMINHO_MICROSLEEP = r"C:\Users\cmay\PycharmProjects\PythonProject\alerta.mp3"

# =========================
# MODELO
# =========================
model = YOLO(CAMINHO_MODELO)

# =========================
# VLC
# =========================
player_fadiga = vlc.MediaPlayer(CAMINHO_FADIGA)
player_acorda = vlc.MediaPlayer(CAMINHO_MICROSLEEP)

audio_fadiga_tocando = False
audio_acorda_tocando = False
manter_acorda_ate = None
ja_entrou_microsleep = False

# =========================
# HAAR CASCADE
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# CONTROLES
# =========================
tempo_inicio_fechado = None
ultimo_beep = 0

LIMITE_PISCADA = 0.4
LIMITE_ATENCAO = 1.0
LIMITE_FADIGA = 2.0
LIMITE_MICROSLEEP = 3.0
TEMPO_ACORDA_APOS_ABRIR = 10.0

ultimo_rosto = None
alpha = 0.65


def classificar_olho(recorte):

    if recorte.size == 0:
        return None, 0.0

    results = model(recorte, verbose=False)
    result = results[0]

    cls = int(result.probs.top1)
    conf = float(result.probs.top1conf)
    nome = model.names[cls]

    return nome, conf


def tocar_fadiga():
    global audio_fadiga_tocando

    if not audio_fadiga_tocando:
        player_fadiga.stop()
        time.sleep(0.05)
        player_fadiga.play()
        audio_fadiga_tocando = True


def parar_fadiga():
    global audio_fadiga_tocando

    player_fadiga.stop()
    audio_fadiga_tocando = False


def tocar_acorda():
    global audio_acorda_tocando

    player_acorda.stop()
    time.sleep(0.05)
    player_acorda.play()
    audio_acorda_tocando = True


def parar_acorda():
    global audio_acorda_tocando

    player_acorda.stop()
    audio_acorda_tocando = False


def suavizar_box(box_atual, box_anterior, alpha=0.65):

    if box_anterior is None:
        return box_atual

    x, y, w, h = box_atual
    xa, ya, wa, ha = box_anterior

    x_s = int(alpha * xa + (1 - alpha) * x)
    y_s = int(alpha * ya + (1 - alpha) * y)
    w_s = int(alpha * wa + (1 - alpha) * w)
    h_s = int(alpha * ha + (1 - alpha) * h)

    return x_s, y_s, w_s, h_s


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_vis = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(80, 80)
    )

    resultado_final = "sem deteccao"
    confianca_final = 0.0
    mensagem_tela = ""

    if len(faces) > 0:

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        x, y, w, h = suavizar_box((x, y, w, h), ultimo_rosto, alpha)
        ultimo_rosto = (x, y, w, h)

        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_y1 = y + int(h * 0.15)
        roi_y2 = y + int(h * 0.55)
        roi_x1 = x + int(w * 0.08)
        roi_x2 = x + int(w * 0.92)

        roi_y1 = max(0, roi_y1)
        roi_y2 = min(frame.shape[0], roi_y2)
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(frame.shape[1], roi_x2)

        olhos_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if olhos_roi.size != 0:

            roi_h, roi_w = olhos_roi.shape[:2]
            metade = roi_w // 2

            olho_esq = olhos_roi[:, :metade]
            olho_dir = olhos_roi[:, metade:]

            classe_esq, conf_esq = classificar_olho(olho_esq)
            classe_dir, conf_dir = classificar_olho(olho_dir)

            if classe_esq is None and classe_dir is None:
                resultado_final = "sem deteccao"
                confianca_final = 0.0

            elif classe_esq is not None and classe_dir is None:
                resultado_final = classe_esq
                confianca_final = conf_esq

            elif classe_esq is None and classe_dir is not None:
                resultado_final = classe_dir
                confianca_final = conf_dir

            else:
                if classe_esq == classe_dir:
                    resultado_final = classe_esq
                    confianca_final = max(conf_esq, conf_dir)
                elif conf_esq >= conf_dir:
                    resultado_final = classe_esq
                    confianca_final = conf_esq
                else:
                    resultado_final = classe_dir
                    confianca_final = conf_dir

            if resultado_final == "olhos_abertos":
                cor = (255, 0, 0)

            elif resultado_final == "olhos_fechados":
                cor = (0, 255, 255)

            else:
                cor = (0, 0, 255)

            texto = (
                f"{resultado_final} {confianca_final:.2f}"
                if resultado_final != "sem deteccao"
                else resultado_final
            )

            cv2.rectangle(frame_vis, (roi_x1, roi_y1), (roi_x2, roi_y2), cor, 2)

            cv2.putText(
                frame_vis,
                texto,
                (roi_x1, max(roi_y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                cor,
                2
            )

    else:
        ultimo_rosto = None

        cv2.putText(
            frame_vis,
            "Rosto nao detectado",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    agora = time.time()

    # =========================
    # LOGICA DE ALERTA
    # =========================
    if resultado_final == "olhos_fechados":

        manter_acorda_ate = None

        if tempo_inicio_fechado is None:
            tempo_inicio_fechado = agora
            ja_entrou_microsleep = False

        tempo_fechado = agora - tempo_inicio_fechado

        cv2.putText(
            frame_vis,
            f"Olhos fechados: {tempo_fechado:.1f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if tempo_fechado < LIMITE_PISCADA:
            mensagem_tela = ""

        elif LIMITE_ATENCAO <= tempo_fechado < LIMITE_FADIGA:
            mensagem_tela = "ATENCAO"

            parar_fadiga()

            if not audio_acorda_tocando and agora - ultimo_beep > 1.0:
                winsound.Beep(1200, 200)
                ultimo_beep = agora

        elif LIMITE_FADIGA <= tempo_fechado < LIMITE_MICROSLEEP:
            mensagem_tela = "FADIGA DETECTADA"

            if not audio_acorda_tocando:
                tocar_fadiga()

        elif tempo_fechado >= LIMITE_MICROSLEEP:
            mensagem_tela = "ACORDA"

            parar_fadiga()

            if not ja_entrou_microsleep:
                tocar_acorda()
                ja_entrou_microsleep = True

    else:
        tempo_inicio_fechado = None
        ultimo_beep = 0
        ja_entrou_microsleep = False
        parar_fadiga()

        if audio_acorda_tocando:

            if manter_acorda_ate is None:
                manter_acorda_ate = agora + TEMPO_ACORDA_APOS_ABRIR

            tempo_restante = manter_acorda_ate - agora

            cv2.putText(
                frame_vis,
                f"Alarme critico: {max(0, tempo_restante):.1f}s",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if agora >= manter_acorda_ate:
                parar_acorda()
                manter_acorda_ate = None

    # =========================
    # MENSAGEM NA TELA
    # =========================
    if mensagem_tela != "":

        if mensagem_tela == "ATENCAO":
            cor_msg = (0, 255, 255)

        elif mensagem_tela == "FADIGA DETECTADA":
            cor_msg = (0, 165, 255)

        else:
            cor_msg = (0, 0, 255)

        cv2.putText(
            frame_vis,
            mensagem_tela,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            cor_msg,
            4
        )

    cv2.imshow("Classificacao de olhos - Haar Cascade + VLC", frame_vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


parar_fadiga()
parar_acorda()
cap.release()
cv2.destroyAllWindows()

print("Desligando...")
