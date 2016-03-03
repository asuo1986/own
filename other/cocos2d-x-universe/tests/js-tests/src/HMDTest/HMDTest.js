

var HMDTestScene = TestScene.extend({
    runThisTest:function () {
        //this.addChild(new PathTestLayer());
        var hmdScene = cc.HMDScene.create();
                                    
        var dirLight = jsb.DirectionLight.create(cc.math.vec3(-1.0, -1.0, 0.0), cc.color(250, 250, 250));
        hmdScene.addNode(dirLight);
                                    
        var ambLight = jsb.AmbientLight.create(cc.color(100, 100, 100));
        hmdScene.addNode(ambLight);
                                    
        var sprite = jsb.Sprite3D.create("HMDTest/teapot.c3b");
        sprite.setScale(0.01);
        sprite.setRotation3D(cc.math.vec3(-90, 180, 0))
        sprite.setPosition3D(cc.math.vec3(0, -0.05, -0.25))
        sprite.setTexture("HMDTest/teapot.png");
        sprite.runAction(
                         cc.rotateBy( 10, cc.math.vec3( 0.0, 360.0, 0.0)).repeatForever()
                         );
        hmdScene.addNode(sprite);
                                    
        var keyboardEventListener = cc.EventListener.create({
          event: cc.EventListener.KEYBOARD,
          onKeyReleased: function(keyCode, event){
            var scene = event.getCurrentTarget();
            var head_rot = scene.getHeadRotationQuat();
            var head_pos = scene.getHeadPosition3D();
                        
            if (keyCode == 65) { // KEY_A
              head_pos.x = head_pos.x + -0.01;
            }
            if (keyCode == 68) { // KEY_D
              head_pos.x = head_pos.x + 0.01;
            }
            if (keyCode == 87) { // KEY_W
              head_pos.z = head_pos.z + -0.01;
            }
            if (keyCode == 83) { // KEY_S
              head_pos.z = head_pos.z + 0.01;
            }
                                                            
            scene.setHeadPosition3D(head_pos);
            
            if (keyCode == 219) { // KEY_LEFT_BRACKET
                scene.setEyeDistanceMillimeters(scene.getEyeDistanceMillimeters() - 1.5);
            }
            if (keyCode == 221) { // KEY_RIGHT_BRACKET
                scene.setEyeDistanceMillimeters(scene.getEyeDistanceMillimeters() + 1.5);
            }
                 
            if (keyCode == 38) { // KEY_UP_ARROW
            head_rot = cc.math.quatMultiply( head_rot, cc.math.quaternion(cc.math.vec3(1,0,0), 0.01))
            }
            if (keyCode == 40) { // KEY_DOWN_ARROW
            head_rot = cc.math.quatMultiply( head_rot, cc.math.quaternion(cc.math.vec3(1,0,0), -0.01))
            }
            if (keyCode == 37) { // KEY_LEFT_ARROW
            head_rot = cc.math.quatMultiply( head_rot, cc.math.quaternion(cc.math.vec3(0,1,0), 0.01))
            }
            if (keyCode == 39) { // KEY_RIGHT_ARROW
            head_rot = cc.math.quatMultiply( head_rot, cc.math.quaternion(cc.math.vec3(0,1,0), -0.01))
            }
                                                    
              scene.setHeadRotationQuat(head_rot);
                                                            

                                                            
                                                            
            
            // Stop propagation, so yellow blocks will not be able to receive event.
            event.stopPropagation();
          }
        });
                             
        cc.eventManager.addListener(keyboardEventListener.clone(), hmdScene);
        director.runScene(hmdScene);
    }
});
