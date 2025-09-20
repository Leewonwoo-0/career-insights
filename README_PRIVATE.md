# 🔒 Private Configuration Guide

## 현재 설정

### 공개될 파일 (Git에 업로드)
✅ **README.md** - 전체 개요
✅ **KNOWLEDGE_MAP.md** - 지식 구조도 (제목만)
✅ **portfolio.md** - 공개 포트폴리오
✅ **.gitignore** - 설정 파일

### 비공개 파일 (Git에서 제외)
🔒 모든 폴더 (01_investment ~ 09_research_papers)
🔒 EXECUTIVE_SUMMARY.md
🔒 PORTFOLIO_NEUROSCIENCE.md
🔒 FINAL_REPORT.md
🔒 KNOWLEDGE_KEYWORDS_INDEX.md

## Git 초기화 명령어

```bash
# 1. Git 초기화
git init

# 2. 공개 파일만 추가
git add README.md
git add KNOWLEDGE_MAP.md
git add portfolio.md
git add .gitignore

# 3. 첫 커밋
git commit -m "Initial commit - Public portfolio structure"

# 4. GitHub 리포지토리 연결
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 5. 푸시
git push -u origin main
```

## 나중에 특정 파일 공개하기

특정 파일을 선택적으로 공개하려면:

```bash
# 강제로 특정 파일 추가 (gitignore 무시)
git add -f 03_loopie_startup/public_overview.md
```

## 보안 체크리스트

- [ ] 민감한 투자 전략 제거 확인
- [ ] 개인정보 제거 확인
- [ ] 비즈니스 기밀 제거 확인
- [ ] 연구 노트 비공개 확인
- [ ] 코딩 학습 자료 비공개 확인

## 공개 가능한 내용 수준

### README.md
- 전체 프로젝트 구조 (O)
- 일반적인 성과 지표 (O)
- 링크 (내부 문서는 404) (O)

### KNOWLEDGE_MAP.md
- 폴더 구조 (O)
- 문서 제목 (O)
- 간단한 설명 (O)
- 상세 내용 (X)

### portfolio.md
- 공개 프로젝트 (O)
- 일반적 기술 스택 (O)
- 구체적 알고리즘 (X)
- 상세 전략 (X)