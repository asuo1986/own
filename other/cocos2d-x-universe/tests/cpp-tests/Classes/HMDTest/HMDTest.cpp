
#include "HMDTest.h"
#include "cocos2d.h"

USING_NS_CC_EXT;
USING_NS_CC;

class TestHMDScene : public HMDScene {
  
public:
  CREATE_FUNC(TestHMDScene);
  
  Sprite3D* sprite;
  
  TestHMDScene() {
  }
  
  void initScene() {

    sprite = Sprite3D::create("HMDTest/teapot.c3b");
    sprite->setScale(0.01f);
    sprite->setRotation3D(Vec3(-90.0f, 180.0f, 0.0f));
    sprite->setPosition3D(Vec3(0.0f, -0.05f, -0.25f));
    sprite->runAction(RepeatForever::create(RotateBy::create(10.0f, Vec3(0.0f, 360.0f, 0.0f))));
    this->addNode(sprite);
    
    auto dir_light = DirectionLight::create(Vec3(-1.0f, -1.0f, 0.0f), Color3B(250, 250, 250));
    this->addNode(dir_light);
    
    auto amb_light = AmbientLight::create(Color3B(200, 200, 200));
    this->addNode(amb_light);
    
    this->scheduleUpdate();
    
    
    auto eventListener = EventListenerKeyboard::create();
    
    eventListener->onKeyPressed = [](EventKeyboard::KeyCode keyCode, Event* event){
      
      HMDScene* scene = static_cast<HMDScene*>(event->getCurrentTarget());
      switch(keyCode){
        case EventKeyboard::KeyCode::KEY_UP_ARROW:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_X, 0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_DOWN_ARROW:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_X, -0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_LEFT_ARROW:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_Y, -0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_RIGHT_ARROW:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_Y, 0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_COMMA:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_Z, -0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_PERIOD:
          scene->setHeadRotationQuat(scene->getHeadRotationQuat() * Quaternion(Vec3::UNIT_Z, 0.01f));
          break;
          
          
        case EventKeyboard::KeyCode::KEY_A:
          scene->setHeadPosition3D(scene->getHeadPosition3D() + Vec3(-0.01f, 0.0f, 0.0f));
          break;
        case EventKeyboard::KeyCode::KEY_D:
          scene->setHeadPosition3D(scene->getHeadPosition3D() + Vec3(0.01f, 0.0f, 0.0f));
          break;
        case EventKeyboard::KeyCode::KEY_W:
          scene->setHeadPosition3D(scene->getHeadPosition3D() + Vec3(0.0f, 0.0f, -0.01f));
          break;
        case EventKeyboard::KeyCode::KEY_S:
          scene->setHeadPosition3D(scene->getHeadPosition3D() + Vec3(0.0f, 0.0f, 0.01f));
          break;
          
        case EventKeyboard::KeyCode::KEY_LEFT_BRACKET:
          scene->setEyeDistanceMillimeters(scene->getEyeDistanceMillimeters() - 1.5f);
          break;
        case EventKeyboard::KeyCode::KEY_RIGHT_BRACKET:
          scene->setEyeDistanceMillimeters(scene->getEyeDistanceMillimeters() + 1.5f);
          break;
          
        default:
          break;
      }
    };
    
    this->_eventDispatcher->addEventListenerWithSceneGraphPriority(eventListener,this);
    
  }
  

  void update(float dt) {
    Scene::update(dt);
  }
  
  
  virtual ~TestHMDScene() {
    
  }
  
};

HMDTests::HMDTests()
{
  ADD_TEST_CASE(HMDTestBasic);
};


//------------------------------------------------------------------
//
// HMD Scene Test
//
//------------------------------------------------------------------
HMDTestBasic::HMDTestBasic()
{
}

HMDTestBasic::~HMDTestBasic()
{
  
}

void HMDTestBasic::onEnter() {
  TestHMDScene* scene = TestHMDScene::create();
  scene->initScene();
  Director::getInstance()->replaceScene(scene);
  
  /**
  auto keyboardEventListener = EventListenerKeyboard::create();
  keyboardEventListener->onKeyReleased = [](EventKeyboard::KeyCode key, Event* event){
    auto hmdScene = static_cast<HMDScene*>(event->getCurrentTarget());
    
    Vec3 rot = hmdScene->getHeadRotation();
    Vec3 pos = hmdScene->getHeadPosition();
    
    if (key == EventKeyboard::KeyCode::KEY_LEFT_ARROW) {
      rot.y += -0.1f;
    }
    if (key == EventKeyboard::KeyCode::KEY_RIGHT_ARROW) {
      rot.y +=  0.1f;
    }
    if (key == EventKeyboard::KeyCode::KEY_UP_ARROW) {
      rot.x += -0.01f;
    }
    if (key == EventKeyboard::KeyCode::KEY_DOWN_ARROW) {
      rot.x +=  0.01f;
    }
    if (key == EventKeyboard::KeyCode::KEY_W) {
      pos.z += -0.01f;
    }
    if (key == EventKeyboard::KeyCode::KEY_S) {
      pos.z +=  0.01f;
    }
    if (key == EventKeyboard::KeyCode::KEY_A) {
      pos.x += -0.01f;
    }
    if (key == EventKeyboard::KeyCode::KEY_D) {
      pos.x +=  0.01f;
    }
    
    hmdScene->setHeadPose(pos, rot);
    event->stopPropagation();
  };
  
  _eventDispatcher->addEventListenerWithSceneGraphPriority(keyboardEventListener->clone(), scene);
   **/
}

std::string HMDTestBasic::title() const
{
    return "HMD Basic Test";
}

std::string HMDTestBasic::subtitle() const
{
    return "Testing the HMD stereo render scene";
}

