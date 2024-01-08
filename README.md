# LLM-Lab
LLM 모델을 바탕으로 Finetuning 수행을 위한 실험 코드입니다.
편리한 실험을 하기 위해 Model, Dataset, Prompt가 변경이 되어도 코드의 수정 비용을 최소화하고 실험에만 올인할 수 있도록 코드를 추상화하였습니다.
실험을 위해서는 run_finetuning_tasks.sh 파일만 작동시키도록 개발을 진행 중입니다.
가급적 모든 파라미터를 조작하며 실험하기 위해 run_finetuning_tasks.sh 에 파라미터를 작성해 두었으며, 파라미터들을 지속적으로 추가할 예정입니다.
실험에 대한 내용은 상업적으로도 무료로 이용이 가능한 MLFlow를 사용했으며, 입력 파라미터의 모든 내용을 MLFlow에 입력하는 코드를 곧 추가할 예정입니다.

현재는 LLM 모델에 LLaMA2, KoALPACA 모델 등이 있으며 TinyLlama 등 저명한 모델들 실험이 필요할 시 지속적으로 작성하고 실험 결과를 공유할 예정입니다.
Finetuning Task 또한 현재는 General 한 한국어 이해를 위한 Finetuning, Q&A Task 만 존재하지만 필요에 따라 다양한 Downstream Task를 추가할 수 있습니다.

## 프로젝트 구조도
LLM_Lab/
│
├── config/                                 # 전체 설정 관련 모듈들을 포함하는 패키지
│   ├── arguments.py                            # 명령줄 인자 파싱
│   ├── config_builder.py                       # 설정 객체 빌드
│   ├── constants.py                            # 상수 값 설정
│   ├── training_environment.py                 # 훈련 환경 설정
│   ├── models/                                 # 모델 관련 설정
│   │   ├── lora_config.py                          # LoRA 설정
│   │   ├── model_config.py                         # 모델 구성 설정
│   │   └── quantization_config.py                  # 양자화 설정
│   └── training/                               # 훈련 설정
│       ├── tokenizer_config.py                     # AutoTokenizer 파라미터 설정
│       ├── training_config_manager.py              # Transformers TrainingArguments 파라미터 설정
│       ├── training_logging_config.py              # 로깅 구성 설정
│       └── training_config.py                      # 훈련 구성 설정
│
├── utils/                                  # 유틸리티 함수 및 클래스
│   ├── directory_utils.py                      # 디렉토리 관련 유틸리티
│   ├── experiment_datetime_utils.py            # 실험 날짜/시간 유틸리티
│   └── os_environment_utils.py                 # OS 환경 관련 유틸리티
│
├── data/                                   # 데이터 로딩 및 처리
│   └── data_loader.py                          
│
├── models/                                 # 모델 로딩 및 처리
│   └── model_loader.py                         
│
├── prompts/                                # 프롬프트 관련
│   └── prompt_loader.py
│
├── run_finetuning_tasks.sh                 # MRC Q&A Task 실행 스크립트
├── run_mlflow.sh                           # MLFlow 실행 스크립트
├── LLM_finetuning_main.py                  # main 모듈
├── LLM_finetuning_main_arguments.py        # main 문에 필요한 arguments
└── requirements.txt                        # 의존성 목록
