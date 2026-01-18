from brakesim.models.wheel import wheel_rhs


def test_wheel_decelerates_from_brake_torque_only():
    omega = 100.0
    T_brake = 50.0
    Fx = 0.0          # no tire force
    Re = 0.25
    Iw = 1.2

    domega = wheel_rhs(omega, T_brake, Fx, Re, Iw)

    assert domega < 0.0


def test_brake_overcomes_tire_reaction():
    omega = 100.0
    T_brake = 200.0   # strong brake torque
    Fx = -500.0       # braking tire force
    Re = 0.25
    Iw = 1.2

    domega = wheel_rhs(omega, T_brake, Fx, Re, Iw)

    assert domega < 0.0

def test_tire_force_can_spin_wheel_up():
    omega = 100.0
    T_brake = 10.0    # weak brake
    Fx = -500.0
    Re = 0.25
    Iw = 1.2

    domega = wheel_rhs(omega, T_brake, Fx, Re, Iw)

    assert domega > 0.0
