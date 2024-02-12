# wav2vec2-conformer Fine-Tuning
KT AIVLE BIG PROJECT에서 수행했던 한국어 AI회화 플랫폼을 위한 STT 모델 파인튜닝

<br>

# 데이터
1. AI HUB 자유대화 음성(일반남여)
2. AI HUB 인공지능 학습을 위한 외국인 한국어 발화 음성 데이터

1, 2번의 validation data에서 랜덤으로 추출하여 총 오디오 50시간 파인튜닝

<br>

# 모델
- 42MARU
/
ko-spelling-wav2vec2-conformer-del-1s 의 모델 파인튜닝

![wav2vec2](https://github.com/dfkjdsfisdjfk/wav2vec2_conformer_fine-tuning/assets/110804423/d81991f0-5913-40ad-88a2-b95d8035eaa9)

<br>

# 성능 개선
- 자체 생성 데이터 셋 에서 wer 30% -> wer 14%
- 모델 허깅페이스 주소 : https://huggingface.co/nooobedd/wav2vec_custom

<br>

# 디렉토리 구조
```
.root
├─ model.py / 모델 학습
├─ requirements.txt
├─ to_hugginface.ipynb / 허깅페이스 업로드
└─ utils
   ├─ G2P / 한국어 음운 변환 라이브러리
   │  ├─ Dic
   │  │  ├─ KoG2PDic.txt
   │  │  ├─ nSheetWords.csv
   │  │  └─ rulebook.txt
   │  ├─ KoG2P.py
   │  ├─ KoG2Padvanced.py
   │  └─ unicode.py
   ├─ make_datasets_conformer.py / 허깅페이스 데이터 셋 생성
   ├─ make_foreign_csv.py / 외국인 음성 csv 생성
   ├─ make_kor_csv.py / 한국인 음성 csv 생성
   ├─ DataCollatorCTCWithPadding.py  / DataCollatorCTCWithPadding 정의
   ├─ final_df_make.py / 최종 csv 생성
   └─ __init__.py

```

# G2P 라이브러리 출처
```
@article{Munetal2022,
  author = {Mun, Seongmin and Kim, Su-Han and Ko, Eon-Suk},
  year = {2022},
  title = {A proposal to improve on existing Grapheme-to-Phoneme conversion models informed by linguistics},
  journal = {The Korean Society for Language and Information},
  volume = 26,
  number = {2},
  pages = {27--46}
}
```
