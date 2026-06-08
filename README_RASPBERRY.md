# Execucao na Raspberry Pi - versao NCNN

Esta versao usa o modelo classificador exportado para NCNN, evitando executar
o `best.pt` com PyTorch/Ultralytics na Raspberry Pi.

## Arquivos principais

```text
train3/codigo
train3/weights/best_ncnn_model/model.ncnn.param
train3/weights/best_ncnn_model/model.ncnn.bin
fadiga.mp3
alerta.mp3
offline_wheels/ncnn-*.whl
```

O script `train3/codigo` deve ser executado com Python:

```bash
python train3/codigo
```

## Preparar o ambiente no pendrive

Considerando que o pendrive esta montado em `/usb`:

```bash
python3 -m venv --system-site-packages /usb/venv
source /usb/venv/bin/activate
python -m pip install --upgrade pip
```

Com internet:

```bash
pip install ncnn --no-cache-dir
```

Sem internet, usando a wheel incluida no repositorio:

```bash
pip install --no-index --no-deps offline_wheels/ncnn-1.0.20250503-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

Dependencias do sistema:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-numpy vlc alsa-utils
```

## Rodar

Na pasta raiz do projeto:

```bash
source /usb/venv/bin/activate
python train3/codigo
```

Pressione `q` para sair.

## Audio na saida P2

As caixas de som devem estar:

- alimentadas pela USB da Raspberry;
- conectadas na saida P2 da Raspberry.

O script tenta selecionar a saida P2 com `amixer`. Se o audio nao sair pela P2,
abra as configuracoes de audio do Raspberry Pi OS e selecione manualmente
`Headphones`/`Analog`.

## Logica de alerta

- abaixo de `0.4s`: piscada normal;
- de `1.0s` a `3.0s`: `ATENCAO` visual, sem audio;
- acima de `3.0s` com olhos fechados continuamente: toca `alerta.mp3`;
- o alerta critico continua por `10s` sem reiniciar por piscadas;
- depois dos `10s`, o som so toca novamente se os olhos fecharem por mais `3s`;
- se o rosto/olhos ficarem sem deteccao, o sistema nao dispara audio.
