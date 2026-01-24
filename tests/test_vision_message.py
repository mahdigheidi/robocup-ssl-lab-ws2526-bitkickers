from team_control.network.receiver import Vision


def test_vision_initialization():
    vision = Vision(10020)
    print(vision.addr)
    assert vision is not None
    data = vision.listen()
    # assert data is not None
    print(data)


test_vision_initialization()
