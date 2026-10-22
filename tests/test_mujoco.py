import mujoco


def test_mujoco_physics():
    xml = """
    <mujoco model="basic_test">
        <option timestep="0.002" gravity="0 0 -9.81"/>

        <worldbody>
            <geom name="floor"
                  type="plane"
                  size="2 2 0.1"/>

            <body name="falling_body" pos="0 0 1">
                <freejoint/>
                <geom name="sphere"
                      type="sphere"
                      size="0.05"
                      mass="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    initial_z = data.qpos[2]

    for _ in range(500):
        mujoco.mj_step(model, data)

    final_z = data.qpos[2]

    assert final_z < initial_z, (
        f"The body did not fall as expected: "
        f"initial_z={initial_z}, final_z={final_z}"
    )
    assert abs(data.time - 1.0) < 1e-9
