
#include "cocos2d.h"
#include "hmd/CCHMDScene.h"
#include "base/CCDirector.h"
#include "2d/CCSprite.h"
#include "3d/CCSprite3D.h"
#include "2d/CCNode.h"


NS_CC_BEGIN

static float s_IPDmm = 64.0f;

HMDScene::HMDScene():
  Scene(),
  _left_cam(NULL),
  _right_cam(NULL)
{
  
  auto s = Director::getInstance()->getWinSize();
  float ratio = (GLfloat)s.width / s.height;
  
  _head_node = Node::create();
  Scene::addChild(_head_node);
  
  _attach_node_left = Node::create();
  _attach_node_right = Node::create();
  
  _left_cam = setupCamera( ratio, true);
  _attach_node_left->addChild(_left_cam);
  _head_node->addChild(_attach_node_left);
  
  _right_cam = setupCamera( ratio, false);
  _attach_node_right->addChild(_right_cam);
  _head_node->addChild(_attach_node_right);
  
  _cam_mask = ((unsigned short)CameraFlag::USER7) | ((unsigned short)CameraFlag::USER8);

  setHeadPosition3D(Vec3::ZERO);
  setHeadRotationQuat(Quaternion(Vec3::UNIT_Z, 0.0f));
  setEyeDistanceMillimeters(s_IPDmm);
}

HMDScene::~HMDScene() {
  
}

Camera* HMDScene::setupCamera(float ratio, bool left) {
  
  auto s = Director::getInstance()->getWinSize();
  auto sizeInpixels = Director::getInstance()->getWinSizeInPixels();
  
  auto fboSize = Size(sizeInpixels.width * 1.f, sizeInpixels.height * 1.f);
  auto fbo = experimental::FrameBuffer::create(1, fboSize.width, fboSize.height);
  auto rt = experimental::RenderTarget::create(fboSize.width, fboSize.height);
  auto rtDS = experimental::RenderTargetDepthStencil::create(fboSize.width, fboSize.height);
  
  fbo->attachRenderTarget(rt);
  fbo->attachDepthStencilTarget(rtDS);
  
  auto sprite = Sprite::createWithTexture(fbo->getRenderTarget()->getTexture());
  sprite->setPosition((left ? s.width * 0.25f : s.width * 0.75f) , s.height * 0.5f);
  sprite->setRotation3D(Vec3(0.0f, 180.0f, 180.0f));
  sprite->setScale(0.5f, 1.0f);
  
  Scene::addChild(sprite);
  
  auto cam = Camera::createPerspective(60, ratio, 0.01f, 100.0f);
  cam->setPosition3D(Vec3(0.0f, 0.0f, 0.0f));
  cam->setCameraFlag(left ? CameraFlag::USER7 : CameraFlag::USER8);
  cam->setDepth(-1);
  cam->setName( (left ? "HMD-Cam-L" : "HMD-Cam-R") );
  cam->setFrameBufferObject(fbo);
  // useful for debugging viewport stuff
  //fbo->setClearColor(Color4F( (left ? 0.25f : 0.0f) , (left ? 0.0f : 0.25f), 0, 1));
  
  return cam;
}

void HMDScene::addNode(Node* node) {
  unsigned short cam_mask = node->getCameraMask();
  cam_mask |= this->getCameraMask();
  node->setCameraMask(cam_mask);
  Scene::addChild(node);
}

Vec3 HMDScene::getHeadPosition3D() const {
  return _head_node->getPosition3D();
}

void HMDScene::setHeadPosition3D(const Vec3& pos) {
  _head_node->setPosition3D(pos);
}

Quaternion HMDScene::getHeadRotationQuat() const {
  return _head_node->getRotationQuat();
}

void HMDScene::setHeadRotationQuat( const Quaternion& rot ) {
  _head_node->setRotationQuat(rot);
}


void HMDScene::setEyeDistanceMillimeters(float dist) {
  s_IPDmm = dist;
  float mmToMeters = (s_IPDmm * 0.001f) * 0.5f;

  Vec3 offset(-mmToMeters, 0.0f, 0.0f);
  
  Mat4 xlate;
  Mat4::createTranslation(offset, &xlate);
  _attach_node_left->setPosition3D(offset);
  
  offset.x *= -1.0f;
  Mat4::createTranslation(offset, &xlate);
  _attach_node_right->setPosition3D(offset);
}

float HMDScene::getEyeDistanceMillimeters() const {
  return s_IPDmm;
}


NS_CC_END