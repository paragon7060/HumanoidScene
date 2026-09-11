# Quest로 움직이며 RL reward 확인

기존 Quest 수집기에 `--rl-reward-debug`만 추가한다. CloudXR Runtime, HTTPS 주소,
Quest CONNECT 절차는 [기존 수집기](QUEST_COLLECTOR_SETUP.md)와 같다.
이 옵션이 없으면 기존 수집 동작은 그대로다. **옵션이 있을 때는 데이터 저장이나
정책 학습 대신 현재 flap-pick RL 환경 1개를 수동 조작한다.**

## 실행

저장소 루트에서 기존과 같은 순서로 실행한다. Runtime/web이 이미 실행 중이면
중복 실행하지 않는다. 일반 Isaac 수집기와 이 검사 모드를 동시에 실행하지 않는다.

```bash
# 터미널 1
./quest_collector.sh runtime
# 터미널 2
./quest_collector.sh web
# Quest에서 기존 HTTPS 페이지 → CONNECT 후, 터미널 3
./quest_collector.sh collect --rl-reward-debug
```

기존 head/wrist 영상 패널 옵션도 유지한다. reward 텍스트와 stereo 장면만으로
가볍게 확인하려면 `--no-quest-camera-overlay --no-camera-preview`를 추가한다.
이 경우 RGB 카메라 센서를 생성하지 않는다.

기존에 `collect_quest_teleop.sh`를 직접 사용했다면 기존 명령 끝에
`--rl-reward-debug`를 추가해도 된다. `XR_RUNTIME_JSON` 등 기존 연결 설정은 필요하다.
체크포인트는 필요 없다. 기존 학습을 중단하거나 재시작하지 않지만, GPU 자원이
부족한 PC에서는 학습과 XR을 동시에 실행하지 않는 것이 좋다.

## 조작

| 입력 | 검사 모드 동작 |
|---|---|
| X / PC C | 시점 보정 후 일시정지 |
| A / PC T | 물리·팔 추종 시작/일시정지, 손 위치·회전 기준 재설정 |
| 오른쪽 검지 트리거 | 오른쪽 gripper 닫기; 놓으면 열기. 기본 오른손 실험은 왼쪽 입력 무시 |
| B / PC R | `quest_ready_02`로 초기화하고 정지 (**녹화 버튼 아님**) |
| Y / PC H | reward 패널 표시/숨김 |

정면을 보고 X → 양 컨트롤러를 편한 자세에 두고 A → 손을 움직여 검사한다.
도달 범위가 부족하면 A로 멈추고 손을 편한 위치로 옮긴 뒤 A로 다시 시작한다.
일시정지 동안 물리·task 시간은 멈추고 VR 화면은 계속 렌더링된다.
추적이 끊겨도 정지하며, 복구 후 A를 눌러야 재개한다.

성공·실패·timeout 때 마지막 reward를 고정 표시한다. RL 내부 자동 reset은 유지하므로
장면은 초기 상태로 돌아가지만 패널은 종료 직전 값이다. B로 새 시도를 준비한다.
초기 안정화 동안은 `SETTLING`으로 표시하고 손 입력을 적용하지 않는다.

## VR 표시값

### 실제 collider·안쪽 면 후보·접촉점·힘 표시

`--rl-reward-debug`에서 `--rl-collision-view`는 기본 ON이다. 기존의 수동 offset 구와
구분되는 표시이며, offset을 입력하지 않아도 실제 형상을 읽는다. 관련 코드는
`rl/debug/collision_overlay.py`, CPU 기하 계산은 `collision_geometry.py`다.

```bash
# collider 검사를 우선할 때 기존 기준점/목표점 4개는 숨길 수 있다.
./quest_collector.sh collect --rl-reward-debug --no-rl-grasp-markers
```

| 색/형태 | 실제 표시 데이터 |
|---|---|
| 청록 윤곽선 | 활성 오른손 두 손가락 collider. mesh는 PhysX cooking convex 결과 |
| 주황 윤곽선 | 두 후보 flap collider. USD Cube는 실제 primitive 크기, mesh는 PhysX cooking 결과 |
| 초록 굵은 면 테두리 + 구 | 상대 손가락 방향을 향한 collider polygon 후보와 그 면적 중심 |
| 보라 상자 윤곽선 | 현재 RL 파지 판정의 bounds + contact margin. top_band 모드면 해당 상단 영역 |
| 빨간 구 + 화살표 | PhysX 센서의 평균 접촉 위치와 해당 손가락–flap 쌍의 법선 접촉 합력 |

**초록 면은 자동 선정된 안쪽 면 후보이며, 실제 패드라는 의미가 확정된 것은 아니다.**
상대 손가락의 collider 중심 방향과 법선이 60° 이내로 정렬된 polygon 중
`면적 × cos(각도)^4`가 가장 큰 것을 선택한다. 오른손 두 손가락 각각 독립적으로 선정한다.
중심은 삼각분할 면적 가중 중심이다. 경계선은 z-fighting을 줄이기 위해 법선 방향으로
0.5mm만 띄워 그리며, 중심 구는 실제 면 위에 둔다. 면 후보는 손가락 자세에 따라 바뀔 수 있다.
이 후보를 보상·파지 판정에 자동 적용하지 않는다.

PhysX cooking 요청은 실행 중인 stage에서 활성 CollisionAPI와 approximation 설정을 읽어
비동기로 수행한다. 원본 visual mesh나 전체 visual bounds를 실제 collider라고 대체하지 않는다.
collider 로컬 형상·내부 scale을 body 좌표계로 변환해 저장하고, 표시 시 live articulation
위치/회전을 적용한다. 지원하지 않는 collider 형식이나 cooking 실패는 경로와 원인을 로그에
남기고 제외한다. `COLLIDERS ready / pending / unavailable`은 그 읽기 상태다.
형상/scale 자체를 편집하면 검사 프로세스를 재시작해야 한다. 표시 중 비활성화된 collider는 숨긴다.

접촉 위치는 **전체 접촉 패치가 아니라 센서의 평균점**이다. 화살표는 평균점에 해당 쌍의
법선 접촉 합력 벡터를 붙여 표시한 것이며 개별 접촉점의 힘이 아니다. 마찰의 접선력·토크는
포함하지 않는다. 힘이 finger에 작용하는 방향을
보여주며 길이는 1cm/N, 최대 12cm로 제한한다. 실제 N 값은 HUD와 기존 진단 로그에서 읽는다.
접촉점이 NaN이거나 힘이 없으면 해당 구/화살표를 숨긴다. 기준점 offset은 사용하지 않는다.

- PC Isaac 창에서 **J**: collision 표시 전체 ON/OFF. **G**: 기존 기준점·목표점 표시.
- **Y**는 reward 패널만 제어한다. HMD 안의 3D 형상 표시는 J/G와 독립적이다.
- `--no-rl-collision-view`로 기능을 비활성화할 수 있다. 양 기능을 모두 꺼도 보상 판정은 같다.
- 3D 형상은 일반 RTX scene geometry이므로 VR 시점에서도 렌더링되는 경로다. 다른 물체를
  투시하는 X-ray는 아니므로 가리면 시점을 옮긴다. 초록 후보가 실제 안쪽 패드인지 먼저 확인한다.
- 자동 reset 직후에는 이전 episode의 접촉점을 새 자세에 그리지 않도록 숨긴다. collider는
  현재 reset 장면을 따른다. 일시정지 중에는 마지막 유효 접촉 상태를 보여준다.
- 표시 갱신은 약 10Hz이며 짧은 접촉을 모두 기록하지 않는다. 접촉 이벤트 저장기가 아니다.

이 기능은 보상·관측·collider·관절·마찰을 변경하지 않는다. 시각화 prim에는 물리를 적용하지
않는다. rendering/cooking 오버헤드는 검사 모드에만 추가된다. CPU 기하 테스트와 문법 확인만
수행했으며, 실제 cooking 완료·VR 가시성은 사용자가 실행 후 확인해야 한다.

### 양팔 손가락 기준점을 드래그해서 맞추기

현재 S200062는 [공통 endeffector_center](ENDEFFECTOR_CENTER.md)가 기본 reward/IK/VR 표시 기준입니다.
flap-pick `reaching`은 파지 전 직전 스텝보다 접근하면 +, 후퇴하면 -, 정지하면 0입니다.
파지 획득/유지 중에는 0이고, 파지를 잃은 첫 스텝에는 현재 거리로 기준만 다시 잡습니다.
`Reach signed progress`, `Reach baseline score`, `Reach tracking`을 함께 확인하세요.
`Reaching last 0.5s | episode`는 모든 제어 스텝의 실제 weighted reaching 보상 합계입니다.
순간값이 0이어도 최근 합계나 누적에 짧은 접근 보상이 남습니다. 양/음은 서로 상쇄될 수 있습니다.
최근 구간은 시뮬레이션 시간 기준이며 pause/terminal에서 보존하고 B/reset으로 초기화합니다.
orientation 등 다른 항목은 기존 규칙입니다. 기존 checkpoint는 reward/관측 의미가 달라 새 학습이 필요합니다.
아래의 편집 모드는 저장 후 재시작할 때 반영하며, 실행 중인 RL 정의를 즉시 바꾸지 않습니다.

**마우스로 위치를 맞추는 작업은 [데스크톱 전용 보정 도구](GRASP_DESKTOP_CALIBRATION.md)를
사용하세요.** 아래는 VR 내부 표시 옵션이며, XR 화면은 일반 데스크톱 편집에 적합하지 않습니다.
데스크톱 도구는 헤드셋 없이 실행하며 동일한 네 기준점 파일을 저장합니다.

저장소 루트에서 실행합니다. 이 모드는 기존 기준점/목표점 마커 대신
양팔의 손가락 기준점 **4개**를 표시합니다. 시뮬레이션 물리나 RL 보상은 변경하지 않습니다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-grasp-calibration \
  --desktop-render --no-rl-collision-view
```

1. Quest A 또는 데스크톱 T로 **일시정지**합니다. Isaac Sim의 Stop 대신 수집기의 pause를 사용합니다.
2. Isaac Sim Stage에서 아래 경로의 **`tip` Xform**을 선택합니다.
   부모 손가락 프레임이나 자식 `marker`, 실제 로봇/collision prim은 이동하지 마세요.

   ```text
   /Visuals/GraspCalibration/
     l_f_finger/tip    왼쪽 f, 주황색
     l_b_finger/tip    왼쪽 b, 분홍색
     r_f_finger/tip    오른쪽 f, 청록색
     r_b_finger/tip    오른쪽 b, 파란색
   ```

3. Viewport의 이동 도구로 각 점을 원하는 손가락 안쪽 접촉 위치에 맞춥니다.
   로컬/월드 이동 모드 모두 사용할 수 있습니다. 회전·스케일은 보정 대상이 아닙니다.
   점이 손가락 내부에 가려져 있으면 Stage에서 `tip`을 선택한 뒤 이동하세요.
4. 입력 필드 편집을 끝내고 Viewport에 키보드 포커스를 둔 뒤 **K**를 누릅니다.
   터미널의 `[GRASP CALIBRATION] Saved ...`로 저장 여부를 확인하세요.
   실행 중에는 저장이 거부되므로 먼저 A/T로 정지해야 합니다.
5. A/T로 재생하여 점이 손가락을 따라가는지 확인합니다. 현재 오른팔 전용 RL 설정에서는
   왼손이 고정되어 있으며, 이 보정 모드가 왼팔 제어를 활성화하지는 않습니다.

네 좌표는 작업 디렉터리 기준 `configs/grasp_reference_points.json`에 각 손가락 링크의
로컬 XYZ(m)로 저장됩니다. K를 다시 누르면 해당 파일을 갱신합니다.
다음에 같은 모드로 실행하면 자동으로 불러옵니다. 별도 파일은
`--rl-grasp-calibration-file configs/my_grasp_points.json`으로 선택합니다.
저장 파일의 로봇 모델이 다르면 불러오기를 거부합니다.
처음 실행하여 저장 파일이 없으면 네 점은 **손가락 링크 원점**에 표시됩니다.
저장 없이 종료한 편집은 사라집니다. USD 씬 저장은 필요하지 않습니다.

G는 네 점 표시를 토글합니다. HUD에도 현재 네 로컬 좌표가 표시됩니다.
S200062의 공통 보정 파일은 다음 실행부터 TCP/reaching/파지 기하에도 적용됩니다.
현재 실행 중인 보상 정의는 편집 중 바뀌지 않습니다. 저장 후 재시작하세요.

### 오른손 기준점과 목표점의 3D 표시

두 손가락 기준점만 먼저 확인하려면 다음 명령을 사용한다. flap 목표점과 collision overlay를
숨기고 청록/파란 구만 남긴다. HUD에는 각 local offset(m)과 두 기준점 간격(mm)이 표시된다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-grasp-markers \
  --no-rl-endeffector-centers --no-rl-grasp-targets --no-rl-collision-view \
  --rl-grasp-finger-offsets 0 0 0 0 0 0
```

S200062에서는 모두 0인 CLI 값 대신 저장된 보정값을 기본으로 불러온다.
보정이 없는 다른 모델에서는 link 원점이다. 로컬 좌표는 손가락 link와 함께
움직이므로, 올바른 점이면 팔을 움직이고 그리퍼를 열고 닫아도 동일한 패드 위치에 붙어 있어야 한다.

`--rl-reward-debug`에서는 기본적으로 4개 marker를 월드 좌표계에 표시한다.
물리·보상·관측에는 들어가지 않으며 일반 수집기와 학습 runner는 생성하지 않는다.

| 표시 | 의미 |
|---|---|
| 청록 구 | `r_f_finger` 기준점 |
| 파란 구 | `r_b_finger` 기준점 |
| 주황 작은 큐브 | f 손가락의 flap 목표점 |
| 분홍 작은 큐브 | b 손가락의 반대 면 목표점 |

**기본 offset=0은 link 원점이지, 검증된 fingertip/패드 중심이 아니다.** 두 구가 실제
안쪽 접촉 패드의 끝에 놓이는지 먼저 확인한다. 손가락별 로컬 offset은 다음 옵션으로 지정한다.
단위 m, 순서는 f의 x/y/z 다음 b의 x/y/z다. 아래 0을 보정값으로 바꾸고 검사 프로세스를 재시작한다.

```bash
./quest_collector.sh collect --rl-reward-debug \
  --no-rl-endeffector-centers \
  --rl-grasp-finger-offsets 0 0 0 0 0 0 \
  --rl-grasp-marker-flap flap_right
```

각 손가락의 로컬 축이 다를 수 있다. CLI에서 지정한 비영점 offset은 표시 전용이며
공통 보정 파일이나 reward TCP를 바꾸지 않는다. 목표점은 두 기준점 중점을 flap 면 범위 안에 clamp한
같은 접선 위치의 반대 면 한 쌍이다. 손가락/면 배정은 두 거리 중 큰 값을 작게 하는 쪽으로 고른다.
`auto`(기본)는 현재 잡고 있는 flap을 유지하고, 미파지 시 쌍 거리 비용이 작은 후보를 고른다.
`flap_right`/`flap_left`는 표시 후보만 고정한다. **새 reach 설계 미리보기이며 기존 reaching
reward 목표점은 아니다.** HUD의 `PREVIEW`에 후보와 두 기준점–목표점 거리를 표시한다.

- X로 시점을 정렬하고 A로 움직인다. 다시 A로 멈춘 뒤 정지한 상태에서 둘러볼 수 있다.
- PC Isaac 창에 초점을 두고 G로 marker만 켜고 끈다. Y는 reward 패널만 켜고 끈다.
- `--no-rl-grasp-markers`로 표시를 끈다. `--rl-grasp-marker-radius 0.008`로 기준 구만 키울 수 있다.
- 실제 3D 도형이라 손가락/판 뒤에 가려질 수 있다. 각도를 바꿔 보거나 offset을 확인한다.
  목표 큐브는 판 두께를 정확히 반영하므로 서로 매우 가깝게 보일 수 있다.
- episode 종료 후 marker는 자동 reset된 현재 장면을 따라간다. 패널에 남은 종료 직전
  reward snapshot과 혼동하지 않는다.

구현은 `rl/debug/grasp_markers.py`다. 코드·CPU 기하 테스트만 수행하며 실제 VR 합성은 사용자가 확인한다.

정면에서 약간 아래, 눈앞 28cm의 머리 고정 텍스트 패널에 모든 활성 reward 항목을
약 10Hz로 표시한다. 카메라 패널은 기존처럼 위쪽에 둔다. 표시기는 reward 텍스트를
불투명 RGBA 이미지로 만들고, 기존 카메라 패널과 같은 XR 레이어·GPU 이미지 경로로
전송한다. 일반 `ui.Label`의 생성 여부만으로 VR 표시 성공을 판단하지 않는다.

- 각 항목: `raw × weight × control_dt`, 즉 **실제 해당 스텝의 보상 기여도**.
- `TOTAL`: 해당 스텝의 전체 보상. `RETURN`: 이번 시도의 누적 보상.
- 상승 높이(cm), 유지 시간(s), 좌/우 파지 판정, 좌/우 flap 목표 거리(cm).
- 정지 중에는 마지막 스텝 값이며 새 보상이 발생하지 않는다.

예를 들어 30Hz에서 lift raw=1, weight=5라면 `lift: +0.16667`이다.
성공 보너스 weight=150은 실제 성공 스텝에 `success: +150.00000`으로 표시한다.
TensorBoard `Episode_Reward/*`의 에피소드 정규화 값과 단위가 다르다.
0.1초보다 짧은 비종료 접촉은 화면 갱신 사이에 지나갈 수 있다. 이 모드는 전체
시계열 저장기가 아니며, 종료 보너스는 별도로 마지막 값을 유지한다.

접근 → 접촉 → 파지 → 들어 올리기 → 유지 순으로 `reaching`, `flap_contact`,
`lift`, `holding`, `stable_grasp`가 변하는지 본다. 밀기·충돌 시에는
`prelift_disturbance`도 함께 확인한다. 현재 기본 preset의 `collision` 보상 항은 제거했고
장애물 힘은 진단용으로만 표시한다. 패널의 `Collision termination/penalty: OFF`로 확인한다.
보상은 양손에 각각 독립적으로
배분되는 것이 아니라 task 전체에 계산되며, 현재 기본 성공 조건은 **오른손 한 손 파지**다.

`orientation`은 오른손이 파지 전 면에서 10cm 이내일 때만 주는 약한 정렬 보상이다
(weight 0.5). 각 flap 진단의 `axisErr`는 두 손가락 link 원점 연결축과 flap 법선의
최소 각도(0~90°)다. 0°는 평행 정렬이며, 실제 손가락 패드 방향과의 일치까지
보장하는 값은 아니다. [닫힘 축 확인 절차](RL_RIGHT_HAND_PICK.md)를 함께 따른다.

### 들어 올렸는데 성공하지 않을 때

패널 **맨 위**의 `CHECK`는 아직 만족하지 않은 성공 전제조건이다. `FAIL`에는
실제 종료 원인(장애물 충돌·낙하·영역 이탈·안착 실패 등)을 별도로 표시한다.

| 표시 | 의미 |
|---|---|
| `grasp` | 지정 오른손 flap 파지가 인정되지 않음 |
| `height` | 안착 기준 상승 높이가 6cm를 넘지 못함 |
| `tilt` | 기울기가 40° 기준을 넘음 |
| `hold` | 파지·높이·기울기 조건의 연속 유지가 0.5초에 미달 |
| `initial_wait` | 초기 0.5초 대기가 끝나지 않음 |

pick 성공에서는 속도와 잔여 접촉력 제한을 사용하지 않는다. 기본
`collision_constraints_enabled=False`에서는 장애물 충돌 종료와 collision 비용도 없다.
True로 다시 켠 경우에만 20N 초과로 실패한다. 낙하, 영역 이탈 등 남아 있는 실패는
성공보다 우선한다. 기본 `reset_settle_timeout=0` 모드는
0.5초 고정 대기 후 기준 높이를 저장하며, 안착 타임아웃으로 실패하지 않는다.

패널은 기존 위치에서 머리 기준 왼쪽으로 5cm 이동했다. 상단에 별도 큰 글씨 영역을 두어
상세 reward/marker 텍스트가 길어져도 실패 이유와 성공 체크리스트 크기가 줄지 않는다.
`FAILED: OBSTACLE COLLISION` 등 실패 이유와 측정 force/기준을 표시하며,
`NEED:`(노랑)는 부족한 조건, `OK:`(초록)는 만족한 조건이다. 높이·기울기·유지 시간은
현재값/목표값을 함께 표시한다. 실패·성공·시간 초과 시 패널을 자동으로 다시 표시한다.
자동 reset 후 장면 대신 **종료 직전 샘플**을 계속 표시하고, B로 새 시도를 준비한다.

`R flap_right`와 `R flap_left` 각각의 `raw`는 현재 샘플의 최초 파지 조건, `held`는 제한된
접촉 유예까지 적용한 유지 판정이다. `region=11`은 두 손가락 접촉점이 모두 허용
영역 안에 있다는 뜻이고, `opp=1`은 양면 배치, `F=a/b N`은 해당 flap 접촉력이다.
`gap=...s`는 **접촉 누락 시간**이며 손가락 간격이 아니다. 유지 중 `raw=0, held=1`은
작은 미끄러짐·힘 변화·짧은 누락 때문에 발생할 수 있다. `slip`은 마지막 유효 접촉 대비
손가락 중점 이동, `open+`는 손가락 간격 증가량(mm)이다. 유효 접촉이 계속 확인되면
원점 이동만으로 해제하지 않으며 기준을 갱신한다. 이 제한은 접촉 누락 유예에만 적용한다.
최초 파지 전에는 둘 다 0이다.

물리가 진행 중이면 상승 여부와 무관하게 터미널에도 `[RL GRASP] blocked=...`
진단을 초당 한 번, 종료 스텝에는 즉시 출력한다. 문제가 계속되면 이 줄을 전달하면 된다.
disturbance는 초기 선반 위치·자세 비용과 운동 비용을 분리했다. 1cm 이상 상승하면
전자만 해제되며, 파지 없이 움직이는 박스에는 운동 비용이 남는다. 상세 식과 유지
판정 허용치는 [flap-pick 정의](RL_FLAP_PICK.md)에 있다.

파지 유지 수정 후에는 검사 프로세스를 재시작한다. 이 수정은 학습과 검사 모드가
공유하는 RL 판정이다. 후보 ID observation과 기본 8-action 계약도 변경되어, 수정 전 checkpoint를
직접 재개하는 대신 새 학습을 시작해야 한다.

## 기존 수집기와 동일한 부분 / RL에 맞추는 부분

- 동일: OpenXR 장치·연결, scaled controller 위치/회전 매핑, position gain,
  URDF IK·arm response·orientation weight, head 위치 기준 시점, render quality,
  XR resolution scale, desktop render 옵션.
- RL 유지: `configs/rl_pick_arms_only.py`, named initial pose, 활성 팔 action 순서(기본 오른팔 7+1)와
  증분 크기, actuator·마찰, 관측·reward·충돌·성공/실패 판정과 제어 주기.
  입력만 정책 대신 Quest/IK에서 만든다. 수집기의 별도 관절 구동값·중력 보상·
  self-collision guard를 추가하지 않는다. 따라서 손의 응답은 일반 수집기와 다를 수 있다.
- base·waist·head는 RL대로 고정하며 stick 이동은 사용하지 않는다. HMD를 돌려
  둘러보는 것은 가능하지만 로봇 head joint를 움직이지는 않는다.
- 기본 `active_arm="right"`에서는 왼팔·왼손도 고정된다. 왼쪽 컨트롤러의 X/Y 버튼은
  시점/패널 제어에 계속 쓰지만 왼팔 IK·왼쪽 trigger action은 생성하지 않는다.
  동작 중 tracking 검사는 HMD와 활성 오른쪽 컨트롤러만 필요하다.
- head/wrist RGB 패널·해상도·desktop camera preview 옵션은 그대로 사용한다.
  이 영상은 표시 전용이고 RL observation에는 들어가지 않는다. Depth·데이터 기록·
  공장 배경은 생성하지 않는다. `--control-hz`, `--episode-seconds`, `--arm-stiffness`,
  `--arm-damping`, recording 관련 옵션은 이 모드에서 적용하지 않는다.
- 최소 구현은 **controllers + scaled**만 지원한다. `--hand-switch`, hands,
  relative/absolute mapping, `--scene-config`, `--domain-randomization`은 거부한다.
  수집기의 `--auto-start`와 무관하게 A로 명시적으로 시작한다.

실제 로봇 안전 제어기가 아니다. USD 물체를 순간 이동시키지 않고 기존 RL action을
통해 물리적으로 움직여 접촉 보상을 검사한다.

## 설정 수정

- `configs/rl_pick_arms_only.py`: 초기 프리셋 이름, 손 선택, 성공 조건과 실험별 설정.
- `src/kuavo_isaaclab_scene/rl/managers/rewards.py`: reward 가중치.
- `src/kuavo_isaaclab_scene/rl/mdp/rewards.py`: 항목별 계산식.

실험별 가중치를 바꾸려면 config의 기존 `configure(env_cfg, agent_cfg)` 안에
`env_cfg.rewards.reaching.weight = 6.0`처럼 추가한다. 학습 중인 config를 바꾸기보다
복사본을 만들어 다음과 같이 선택하는 것을 권장한다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-config configs/my_reward_probe.py
```

설정 수정은 **검사 프로세스를 재시작**해야 반영된다. 자동 파일 reload는 없다.
로컬 초기 자세·gripper config를 읽으므로 기존 학습의 `manifest.json`/`env.yaml`과
다르면 그 checkpoint가 사용한 환경의 완전한 재현은 아니다.

구현은 `rl/debug/`에 격리되어 있고 학습 runner에서는 import하지 않는다.
RewardProbe는 reward 계산 후, 자동 reset 전에 이미 계산된 값을 읽을 뿐
RewardManager.compute를 추가 호출하지 않는다. 변경 후 시뮬레이터 실행/VR 시각 검증은
수행하지 않았으며 실제 패널 위치·크기와 조작성은 사용자가 확인한다.

## 패널이 안 보이거나 오류 로그가 나오는 경우

- Y/H를 누르면 `[RL REWARD] Reward panel ON/OFF`와 `[XR REWARD TEXTURE]`가 출력된다.
  `ON`, `layer=True`, `panel=True`, `widget_ready=True`인데도 안 보이면 시점 방향이나
  가림 문제일 수 있다. 먼저 카메라 패널을 끄고 확인하고, 필요하면 기존
  `--xr-overlay-forward-axis +z`로 방향을 비교한다. 이는 원인이 확정된다는 뜻은 아니다.
- `widget_ready=False`가 계속 나오면 XR UI 생성 문제다. 이 상태 로그와 먼저 발생한
  오류를 함께 전달한다. 버튼 수신 로그만으로 화면 표시 성공을 판단하지 않는다.
- `[XR REWARD TEXTURE] uploads=...`는 GPU 이미지 업로드 횟수이며, 예전 `[XR PANEL]`
  로그만 나온다면 변경 전 프로세스다. 수집기를 재시작해야 한다. 업로드 성공 역시
  실제 헤드셋 합성까지 검증한 값은 아니다. 투명하지 않은 테두리/배경까지 전혀
  보이지 않으면 이 로그와 카메라 패널의 표시 여부를 함께 전달한다.
- RGBA 텍스트 생성에는 Pillow 10.1 이상을 사용한다(현재 로컬 Isaac 환경에 설치됨).
  다른 PC에서는 프로젝트의 `rl` extra에 포함된다. RGB 카메라 센서를 끈 상태에서도
  작은 desktop viewport 업데이트를 유지해 UI 텍스처가 갱신되도록 한다.
- `bind_event_generator ... token: press`가 `[Error]`로 출력되어도 이후
  `[BUTTON] ... pressed`가 나오면 해당 버튼 콜백은 동작한 것이다. 이 메시지만으로
  연결 실패나 프로세스 중단으로 판단하지 않으며, SDK 로그를 전역으로 숨기지 않는다.
- `Body must be non-kinematic`는 무시할 로그가 아니다. 검사 모드의 고정 conveyor
  표면에 대한 불필요한 zero-velocity reset을 차단했다. 박스의 동역학이나 RL 학습
  runner는 변경하지 않는다. 같은 오류가 계속되면 다른 호출/물체도 확인해야 한다.
