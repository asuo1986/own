#ifndef __HMD__SCENE

#define __HMD__SCENE

#include "math/CCMath.h"
#include "2d/CCScene.h"

namespace cocos2d {
  
  class Node;
  class Camera;
  
class CC_DLL HMDScene : public Scene
{
  public:
    CREATE_FUNC(HMDScene);

    /** Set the distance between the left and right "eyes"
     * @param dist The distance in millimeters
     * @js NA
     */
    void setEyeDistanceMillimeters(float dist);
    float getEyeDistanceMillimeters() const;

  
  
    /** set the position of the head (eyes will update)
     *
     * @param pos Position in world space.
     * @js NA
     */
    void setHeadPosition3D(const Vec3& pos);
    Vec3 getHeadPosition3D() const;
  
  
    /** set the rotation of the head (eyes will update)
     *
     * @param rot Rotaion in world space.
     * @js NA
     */
    void setHeadRotationQuat(const Quaternion& rot);
    Quaternion getHeadRotationQuat() const;
  
  
    /** get the full camera mask. This is required to render in both "eyes"
     *
     * @js NA
     */
    inline unsigned short getCameraMask() const { return _cam_mask; }

    /** Add child to be rendered in the stereo view
     *
     * @param node Node to add.
     * @js NA
     */
    void addNode(Node* node);

  CC_CONSTRUCTOR_ACCESS:
    HMDScene();
    virtual ~HMDScene();

  protected:
    unsigned short _cam_mask;
    Camera*     _left_cam;
    Camera*     _right_cam;
    Node*       _head_node;
    Node* _attach_node_left;
    Node* _attach_node_right;
  
  private:
    Camera* setupCamera( float ratio, bool left);
  
  };
} //namespace cocos2d

#endif // __HMD__SCENE