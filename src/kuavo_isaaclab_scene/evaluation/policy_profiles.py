"""Policy schema identifiers, safe to import before starting Isaac Sim."""

DEFAULT_POLICY_PROFILE = "default"
ARM_CLAW_PROFILE = "kuavo-arm-claw"
RWH_KUAVO_V2_S56_PROFILE = "rwh-kuavo-v2-s56"
ARM_CLAW_PROFILES = (ARM_CLAW_PROFILE, RWH_KUAVO_V2_S56_PROFILE)
POLICY_PROFILES = (DEFAULT_POLICY_PROFILE, *ARM_CLAW_PROFILES)
