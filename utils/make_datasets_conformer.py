from datasets import Dataset, Audio
import IPython.display as ipd
import pandas as pd
from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import jamo
import librosa

# 모델에 넣기 위한 음성 데이터 전처리 함수
def prepare_dataset(batch):
    audio = batch["audio"]
    raw, _ = librosa.load(batch['audio'], sr=16000)
    batch["input_values"] = processor(raw,sampling_rate=16000).input_values[0]
    # batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcripts"]).input_ids
    return batch


if __name__ =="__main__":
    # 특징 추출기 정의 및 저장
    feature_extractor = AutoFeatureExtractor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
    feature_extractor.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_conformer')

    # 토크나이저 정의 및 저장
    tokenizer = AutoTokenizer.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
    tokenizer.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_tokenizer')

    # 프로세서 설정
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    
    # final_dataset 불러와서 사용할 데이터 양을 줄이고 음소 기반 모델링을 텍스트 전처리 후 데이터 셋 저장
    datasets_df = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/final_dataset_df.csv', encoding='utf-8')   
    _, datasets_df = train_test_split(datasets_df, stratify=datasets_df['country'], test_size = 0.1, random_state=23)
    datasets_df['text'] = datasets_df['text'].map(lambda x : ''.join(list(jamo.hangul_to_jamo(x))))
    datasets_df.to_csv('./used_dataset_df.csv', index=False, encoding='utf-8')


    # 허깅페이스 데이터 셋에 사용할 컬럼들 설정
    audio_list = list(datasets_df['audio_path'])
    txt = list(datasets_df['text'])
    country = list(datasets_df['country'])

    # 허깅페이스 데이터 셋 설정 및 나라 별 균등 셋 분리
    ds = Dataset.from_dict({'audio' : audio_list,
                        "transcripts": txt,
                        "country" : country})
    ds = ds.class_encode_column("country")
    ds = ds.train_test_split(test_size=0.1, shuffle=True, stratify_by_column='country')
    
    # 데이터 셋 음성 전처리 진행 및 저장
    ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"])
    ds.save_to_disk('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/pre_datasets')

    print('토탈 시간 ', datasets_df['record_time'].sum())
    print('-------------finish-------------')