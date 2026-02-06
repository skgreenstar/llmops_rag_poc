# Operator Manual (운영자 매뉴얼)

이 문서는 프로젝트 운영, 유지보수, 문제 해결을 위한 가이드입니다.

## 1. 시작하기 (Getting Started)

### 사전 요구사항
- Docker & Docker Compose
- Python 3.10+
- Poetry (의존성 관리)

### 설치 및 실행

1. **의존성 설치**
   ```bash
   poetry install
   ```

2. **환경 변수 설정**
   `.env.example`을 복사하여 `.env` 파일을 생성하고 필요한 값을 설정하세요.
   ```bash
   cp .env.example .env
   ```

3. **서비스 시작**
   전체 서비스(Docker 컨테이너, 백엔드, UI)를 한 번에 시작하려면 시작 스크립트를 사용하세요.
   ```bash
   ./start_all.sh
   ```
   이 스크립트는 다음을 수행합니다:
   - Docker 컨테이너 실행 (Dify, DB 등)
   - FastAPI 백엔드 실행 (Port 8000)
   - Streamlit UI 실행 (Port 8501)

## 2. 프로젝트 구조 (Project Structure)

```
.
├── app/                  # 애플리케이션 소스 코드
├── dify/                 # Dify 설정 및 Docker Compose 파일
│   ├── unused_patches/   # 사용되지 않는 패치 파일 보관소
│   └── docker-compose.yaml
├── scripts/              # 유지보수 및 유틸리티 스크립트
├── start_all.sh          # 전체 서비스 시작 스크립트
└── OPERATOR_MANUAL.md    # 본 매뉴얼
```

## 3. 유지보수 스크립트 (Maintenance Scripts)

`scripts/` 디렉토리에 있는 유틸리티 스크립트를 사용하여 상태를 점검하거나 데이터를 관리할 수 있습니다.
모든 스크립트는 프로젝트 루트에서 모듈로 실행하거나 직접 실행할 수 있습니다.

### UI 기능 동기화 (Frontend Sync)
- **프롬프트 목록**: Langfuse API`(/api/public/prompts)`를 통해 실시간으로 프롬프트를 조회합니다.
- **데이터셋 동기화**: Streamlit의 **"📊 데이터셋 관리"** 탭에서 로컬 JSON을 Langfuse Dataset으로 업로드할 수 있습니다.

### Langfuse 연결 확인
Langfuse 서버와의 연결 상태 및 프롬프트 목록을 확인합니다.
```bash
python scripts/check_langfuse_list.py
```

### Qdrant 연결 확인
Qdrant 벡터 데이터베이스 연결 상태를 확인합니다.
```bash
python scripts/check_qdrant.py
```

### Qdrant 초기화
Qdrant의 `knowledge_base` 컬렉션을 삭제합니다. (주의: 데이터가 삭제됩니다)
```bash
python scripts/reset_qdrant.py
```

### 4. 자동화된 평가 (Automated Evaluation)

새로 구현된 **Automated Eval** 기능을 사용하여 Agent의 답변 품질(Faithfulness, Relevance)을 평가하고 Langfuse에 점수를 기록할 수 있습니다.

#### 방법 A: Streamlit UI 사용 (권장)
1. 브라우저에서 `http://localhost:8501` 접속
2. **"✅ 평가 실행"** 탭 클릭
3. **"평가 시작"** 버튼 클릭
4. 결과는 Langfuse 대시보드에서 실시간 확인 가능

#### 방법 B: CLI 실행
```bash
python scripts/run_eval.py
```

- **Faithfulness**: 답변이 Context에 기반했는지 (Hallucination 여부)
- **Relevance**: 답변이 사용자의 질문에 적절한지
- **결과 확인**: Langfuse 대시보드의 Traces 탭에서 각 Trace에 연결된 Scores를 확인할 수 있습니다.

## 5. 문제 해결 (Troubleshooting)

### 포트 충돌
`start_all.sh` 실행 시 포트 충돌 에러가 발생하면, 해당 포트를 사용 중인 프로세스를 종료하거나 Docker 컨테이너를 재시작하세요.
- API: 8000
- UI: 8501
- Dify Web: 80
- Dify API: 5001

### 모듈 임포트 에러
스크립트 실행 시 `ModuleNotFoundError: No module named 'app'` 에러가 발생하면 `PYTHONPATH`를 설정하거나 스크립트 자체가 루트 경로를 인식하도록 수정되었는지 확인하세요. 현재 수정된 스크립트는 자동으로 상위 경로를 참조합니다.

## 5. 기타 (Others)
- **Unused Patches**: `dify/unused_patches/` 폴더에는 현재 `docker-compose.yaml`에서 참조하지 않는 패치 파일들이 보관되어 있습니다. 필요 시 참조하거나 삭제할 수 있습니다.
