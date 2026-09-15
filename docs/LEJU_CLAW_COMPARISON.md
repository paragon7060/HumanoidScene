# S200062 claw와 OpenLET Challenge Cup S52 gripper 비교

확인일: 2026-09-15. OpenLET/leju-kuavo-challenge-cup-2026 master 커밋
`51b3defaf8c032957647c7aa193d1fa20daef1f3`의 소스를 직접 조회했다.

## 결론

양쪽 모두 두 손가락 opening/closing 제어를 제공하지만 동일한 모델이 아니다.
대회 S52의 URDF mesh와 MJCF model/class는 Robotiq 2F-85 계열이다.
`/control_robot_leju_claw` 서비스 명칭은 실제 Leju claw API에 대한 호환
인터페이스이며, 기구학이나 mesh가 실제 Leju claw와 동일하다는 뜻은 아니다.
현재 독립 패키지는 요청대로 S200062에서 추출했고 대회 S52 모델로 교체하지 않았다.

## URDF 비교 — 한 손 기준

| 항목 | S200062-derived 독립 claw | 대회 S52 gripper |
|---|---|---|
| housing / jaw mesh | `l/r_twofinger_base`, `*_bar_[1-4]`, `*_finger` | `2f85/robotiq_85_base_link_fine_new`, outer/inner finger/knuckle |
| hand 자체 링크 | 11 + D405 branch 3 = 14 | gripper branch 9 (D405 별도) |
| revolute joints | bar_1/3/4 × 2 jaws = 6 | outer knuckle × 2 = 2 |
| 연동 | 두 bar_1에 반대 부호 target; 네 passive hinge와 USD closure | master `left/right_gripper_joint`, 반대 knuckle은 multiplier=1 mimic |
| revolute axis | local Y `(0,1,0)` | local X `(1,0,0)` |
| URDF limit | 각 bar hinge `[-0.698,0.698]` rad | 두 knuckle `[0,0.8]` rad |
| 사용 driver command | f `[-0.25,0]`, b `[0,0.25]` rad | master `[0,0.8]` rad |
| 왼손 wrist mount xyz(m) | `(0,-0.0005,-0.041)` | `(0,0,-0.0038)` |
| 왼손 wrist mount rpy(rad) | `(0,0,0)` | `(3.14159,0,1.5708)` |

위 local-axis 비교는 각 링크 프레임 기준이다. 축 차이 하나만으로 다른 기구라
단정한 것은 아니며, mesh 세트·joint topology·origin·range 차이를 함께 확인했다.
렌더링하거나 형상 정합을 수행한 결과는 아니다.

## 대회 URDF와 실제 MuJoCo 모델도 구분해야 한다

- [S52 URDF](https://gitcode.com/OpenLET/leju-kuavo-challenge-cup-2026/blob/51b3defaf8c032957647c7aa193d1fa20daef1f3/src/challenge_cup_simulator/models/biped_s52/urdf/biped_s52.urdf)는
  두 revolute와 여러 fixed joints 및 mimic으로 단순화되어 있다.
- [S52 MJCF](https://gitcode.com/OpenLET/leju-kuavo-challenge-cup-2026/blob/51b3defaf8c032957647c7aa193d1fa20daef1f3/src/challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml)는
  `2f85` class로 driver/coupler/follower/spring_link 8 hinges/hand,
  tendon, equality/connect closure를 사용한다. actuator ctrlrange는 `[0,255]`다.
  URDF만 복사한다고 이 모델의 동역학이 재현되는 것은 아니다.
- [sim_leju_claw_interface.py](https://gitcode.com/OpenLET/leju-kuavo-challenge-cup-2026/blob/51b3defaf8c032957647c7aa193d1fa20daef1f3/src/challenge_cup_simulator/scripts/sim_leju_claw_interface.py)는
  master driver 위치를 `position/0.8*100`으로 변환하여 Leju API의 `[0,100]`
  퍼센트 상태를 제공한다.

따라서 S200062와 대회 S52의 joint action, passive reset, TCP, mount 및
collision/material 설정을 그대로 공유해서는 안 된다. 추후 대회 모델을
별도 preset으로 추가한다면 S200062-derived asset과 출처 및 이름을 분리한다.
