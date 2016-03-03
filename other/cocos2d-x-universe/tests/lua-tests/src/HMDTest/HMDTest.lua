

RunHMDScene = function()
   
end


function quat_mul( lhs, rhs )
  -- WHY is there no cc.math.quantMultiply???
  local x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y
  local y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x
  local z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w
  local w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z
  return cc.quaternion(x, y, z, w)
end

function HMDTestMain()
  cclog("HMDTestMain")

  local scene = cc.HMDScene:create()

  -- set up some scene lights
  local dir_light = cc.DirectionLight:create(cc.vec3(-1.0, -1.0, 0.0), cc.c3b(250, 250, 250))
  scene:addNode(dir_light)

  local amb_light = cc.AmbientLight:create(cc.c3b(100, 100, 100))
  scene:addNode(amb_light)

  -- test a 3D sprite
  local sprite3D = cc.Sprite3D:create("ccb/HMDTest/teapot.c3b")
                      :setScale(0.01)
                      :setRotation3D(cc.vec3(-90, 180, 0)) -- turn right side up
                      :setPosition3D(cc.vec3(0, -0.05, -0.25))
                      :setTexture("ccb/HMDTest/teapot.png")
  sprite3D:runAction(cc.RepeatForever:create(
                              cc.RotateBy:create(10, cc.vec3(0, 360, 0))
                            )
                          )
  scene:addNode(sprite3D)

  local function onKeyReleased(keyCode, event)
    local scene = event:getCurrentTarget()

    local head_rot = scene:getHeadRotationQuat();
    local head_pos = scene:getHeadPosition3D();

    if keyCode == cc.KeyCode.KEY_UP_ARROW then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(1,0,0), 0.01))
      scene:setHeadRotationQuat(head_rot);
    end
    if keyCode == cc.KeyCode.KEY_DOWN_ARROW then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(1,0,0), -0.01))
      scene:setHeadRotationQuat(head_rot);
    end
    if keyCode == cc.KeyCode.KEY_LEFT_ARROW then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(0,1,0), 0.01))
      scene:setHeadRotationQuat(head_rot);
    end
    if keyCode == cc.KeyCode.KEY_RIGHT_ARROW then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(0,1,0), -0.01))
      scene:setHeadRotationQuat(head_rot);
    end
    if keyCode == cc.KeyCode.KEY_COMMA then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(0,0,1), -0.01))
      scene:setHeadRotationQuat(head_rot);
    end
    if keyCode == cc.KeyCode.KEY_PERIOD then
      head_rot = quat_mul(head_rot, cc.quaternion_createFromAxisAngle(cc.vec3(0,0,1), 0.01))
      scene:setHeadRotationQuat(head_rot);
    end


    if keyCode == cc.KeyCode.KEY_A then
      head_pos.x = head_pos.x + -0.01
      scene:setHeadPosition3D(head_pos);
    end
    if keyCode == cc.KeyCode.KEY_D then
      head_pos.x = head_pos.x + 0.01
      scene:setHeadPosition3D(head_pos);
    end
    if keyCode == cc.KeyCode.KEY_W then
      head_pos.z = head_pos.z + -0.01
      scene:setHeadPosition3D(head_pos);
    end
    if keyCode == cc.KeyCode.KEY_S then
      head_pos.z = head_pos.z + 0.01
      scene:setHeadPosition3D(head_pos);
    end

    if keyCode == cc.KeyCode.KEY_LEFT_BRACKET then
      scene:setEyeDistanceMillimeters(scene:getEyeDistanceMillimeters() - 1.5);
    end
    if keyCode == cc.KeyCode.KEY_RIGHT_BRACKET then
      scene:setEyeDistanceMillimeters(scene:getEyeDistanceMillimeters() + 1.5);
    end
  end

  local listener = cc.EventListenerKeyboard:create()
  listener:registerScriptHandler(onKeyReleased, cc.Handler.EVENT_KEYBOARD_RELEASED )

  local eventDispatcher = cc.Director:getInstance():getEventDispatcher()
  eventDispatcher:addEventListenerWithSceneGraphPriority(listener, scene)


  return scene

end
