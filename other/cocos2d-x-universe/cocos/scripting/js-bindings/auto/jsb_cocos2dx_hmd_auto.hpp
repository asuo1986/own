#include "base/ccConfig.h"
#ifndef __cocos2dx_hmd_h__
#define __cocos2dx_hmd_h__

#include "jsapi.h"
#include "jsfriendapi.h"

extern JSClass  *jsb_cocos2d_HMDScene_class;
extern JSObject *jsb_cocos2d_HMDScene_prototype;

bool js_cocos2dx_hmd_HMDScene_constructor(JSContext *cx, uint32_t argc, jsval *vp);
void js_cocos2dx_hmd_HMDScene_finalize(JSContext *cx, JSObject *obj);
void js_register_cocos2dx_hmd_HMDScene(JSContext *cx, JS::HandleObject global);
void register_all_cocos2dx_hmd(JSContext* cx, JS::HandleObject obj);
bool js_cocos2dx_hmd_HMDScene_addNode(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_getHeadRotationQuat(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_setHeadRotationQuat(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_getCameraMask(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_getHeadPosition3D(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_setHeadPosition3D(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_create(JSContext *cx, uint32_t argc, jsval *vp);
bool js_cocos2dx_hmd_HMDScene_HMDScene(JSContext *cx, uint32_t argc, jsval *vp);

#endif // __cocos2dx_hmd_h__
