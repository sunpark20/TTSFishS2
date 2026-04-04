#!/usr/bin/env python3
"""소설 장면 캐릭터별 음성 생성 (0.6B Base 모델)
1.7B CustomVoice와 비교하기 위한 0.6B 버전.
"""

import os
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
OUTPUT_DIR = "output/novel_scene_06b"
SAMPLE_RATE = 24000

# 캐릭터별 음성 + 감정 설정
CHARACTERS = {
    "시오크": {
        "instruct": "30대 남성, 차분하고 낮은 목소리, 약간 비꼬는 듯한 여유로운 톤으로 말합니다.",
        "speed": 0.95,
    },
    "베로시": {
        "instruct": "50대 남성, 굵고 위엄 있는 장군의 목소리, 권위적이고 무게감 있는 톤으로 말합니다.",
        "speed": 0.9,
    },
    "포로": {
        "instruct": "40대 남성, 거칠고 쉰 목소리, 분노에 찬 거친 톤으로 소리칩니다.",
        "speed": 1.1,
    },
    "나레이션": {
        "instruct": "중성적이고 차분한 목소리로, 감정 없이 또박또박 읽습니다.",
        "speed": 1.0,
    },
}

# (순번, 캐릭터, 감정 인스트럭션, 대사)
LINES = [
    (1, "나레이션", "차분하고 담담하게",
     "조금 전의 희극적인 장면 때문에 시오크는 갑작스럽게 화를 내기 어려웠다. 그래서 가벼운 어조로 말했다."),

    (2, "시오크", "가볍고 여유로운 어조로, 살짝 비꼬듯이, 느긋하게",
     "내 아버지께서 불초에게 그런 이름을 주셨지요. 그런데 두억시니 장군입니까?"),

    (3, "나레이션", "담담하게",
     "베로시는 그 질문에 의도적으로 대답하지 않았다."),

    (4, "베로시", "위엄 있고 묵직하게, 약간 비웃는 듯한 톤으로, 느리고 무겁게",
     "믿기 어렵군. 시오크 지울비가 자기 발로 걸어 들어와 붙잡히다니. 내가 알기로 자네는 교의로 붙잡히는 버릇이 있지. 혹 그 버릇이 도진 건가?"),

    (5, "나레이션", "차분하게",
     "베로시가 말한 것은 페로그리미의 일이었다. 시오크는 그것이 사실이었으면 좋겠다고 생각했다."),

    (6, "시오크", "담담하면서도 약간 방어적인 톤으로",
     "설마 진짜 그렇게 믿는 것은 아니겠지요."),

    (7, "나레이션", "짧고 담담하게",
     "베로시는 고개를 끄덕였다."),

    (8, "베로시", "조롱하듯 느리고 위압적으로, 한 단어 한 단어 무겁게",
     "맞아. 나는 자네가 의도에 반하여 붙잡혔다고 생각해야겠군. 그런데 그 경우 자네는 11만 대군 속으로 걸어 들어와 가축 도둑질이나 하다가 붙잡히는 얼간이가 되는데."),

    (9, "나레이션", "약간 긴장감 있게",
     "시오크는 그 말을 인정하는 곤욕을 겪지 않았다. 그의 곁에서 무시무시한 눈으로 베로시를 쏘아보던 포로가 고함을 빽 질렀기 때문이다."),

    (10, "포로", "분노에 차서 소리를 지르듯, 격렬하고 거칠게, 최대한 크게",
     "산양 학살자!"),

    (11, "나레이션", "담담하게",
     "시오크는 어깨를 으쓱였다. 베로시는 자신을 바라보는 포로를 마주 보다가 다시 시오크에게 고개를 돌렸다."),

    (12, "베로시", "진지하고 의문을 품은 톤으로, 약간 놀란 듯",
     "정말로 그런 이유에서?"),

    (13, "나레이션", "차분하게 마무리하듯",
     "시오크는 대답하려 했지만 산양 학살자에 대한 무서운 욕설이 계속되고 있었기에 말소리가 묻힐 것 같았다. 그래서 고개만 끄덕였다. 베로시는 등받이에 몸을 기대곤 관자놀이를 긁적였다."),
]


def generate_line(model, seq, character, emotion, text, output_dir):
    char_info = CHARACTERS[character]
    # instruct 파라미터로 캐릭터 음색 + 감정 결합
    instruct = f"{char_info['instruct']} {emotion}."
    speed = char_info["speed"]

    print(f"  [{seq:02d}] {character} ({emotion})")
    print(f"       \"{text[:40]}...\"" if len(text) > 40 else f"       \"{text}\"")

    results = list(model.generate(
        text=text,
        instruct=instruct,
        speed=speed,
        temperature=0.9,
        top_k=50,
    ))
    audio = np.array(results[0].audio)

    filename = f"{seq:02d}_{character}.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"       -> {filepath}")
    return filepath, audio


def merge_audio(audio_files, output_path):
    silence = np.zeros(int(SAMPLE_RATE * 0.8))
    combined = []
    for _, audio in audio_files:
        combined.append(audio)
        combined.append(silence)
    merged = np.concatenate(combined)
    sf.write(output_path, merged, SAMPLE_RATE)
    print(f"\n  전체 합본: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"  모델 로딩: {MODEL_ID}")
    model = load_model(MODEL_ID)
    print("  모델 로드 완료!\n")

    audio_files = []
    for seq, character, emotion, text in LINES:
        filepath, audio = generate_line(model, seq, character, emotion, text, OUTPUT_DIR)
        audio_files.append((filepath, audio))

    merge_path = os.path.join(OUTPUT_DIR, "full_scene.wav")
    merge_audio(audio_files, merge_path)

    duration = sum(len(a) for _, a in audio_files) / SAMPLE_RATE
    print(f"\n  완료! {len(LINES)}개 대사, 총 {duration:.1f}초")
    print(f"  출력: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
