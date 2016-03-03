#include "jsb_cocos2dx_hmd_auto.hpp"
#include "cocos2d_specifics.hpp"
#include "CocosHMD.h"

template<class T>
static bool dummy_constructor(JSContext *cx, uint32_t argc, jsval *vp) {
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    JS::RootedValue initializing(cx);
    bool isNewValid = true;
    JS::RootedObject global(cx, ScriptingCore::getInstance()->getGlobalObject());
    isNewValid = JS_GetProperty(cx, global, "initializing", &initializing) && initializing.toBoolean();
    if (isNewValid)
    {
        TypeTest<T> t;
        js_type_class_t *typeClass = nullptr;
        std::string typeName = t.s_name();
        auto typeMapIter = _js_global_type_map.find(typeName);
        CCASSERT(typeMapIter != _js_global_type_map.end(), "Can't find the class type!");
        typeClass = typeMapIter->second;
        CCASSERT(typeClass, "The value is null.");

        JS::RootedObject proto(cx, typeClass->proto.get());
        JS::RootedObject parent(cx, typeClass->parentProto.get());
        JS::RootedObject _tmp(cx, JS_NewObject(cx, typeClass->jsclass, proto, parent));
        
        args.rval().set(OBJECT_TO_JSVAL(_tmp));
        return true;
    }

    JS_ReportError(cx, "Constructor for the requested class is not available, please refer to the API reference.");
    return false;
}

static bool empty_constructor(JSContext *cx, uint32_t argc, jsval *vp) {
    return false;
}

static bool js_is_native_obj(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    args.rval().setBoolean(true);
    return true;    
}
JSClass  *jsb_cocos2d_HMDScene_class;
JSObject *jsb_cocos2d_HMDScene_prototype;

bool js_cocos2dx_hmd_HMDScene_addNode(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    bool ok = true;
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_addNode : Invalid Native Object");
    if (argc == 1) {
        cocos2d::Node* arg0;
        do {
            if (args.get(0).isNull()) { arg0 = nullptr; break; }
            if (!args.get(0).isObject()) { ok = false; break; }
            js_proxy_t *jsProxy;
            JSObject *tmpObj = args.get(0).toObjectOrNull();
            jsProxy = jsb_get_js_proxy(tmpObj);
            arg0 = (cocos2d::Node*)(jsProxy ? jsProxy->ptr : NULL);
            JSB_PRECONDITION2( arg0, cx, false, "Invalid Native Object");
        } while (0);
        JSB_PRECONDITION2(ok, cx, false, "js_cocos2dx_hmd_HMDScene_addNode : Error processing arguments");
        cobj->addNode(arg0);
        args.rval().setUndefined();
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_addNode : wrong number of arguments: %d, was expecting %d", argc, 1);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_getHeadRotationQuat(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_getHeadRotationQuat : Invalid Native Object");
    if (argc == 0) {
        cocos2d::Quaternion ret = cobj->getHeadRotationQuat();
        jsval jsret = JSVAL_NULL;
        jsret = quaternion_to_jsval(cx, ret);
        args.rval().set(jsret);
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_getHeadRotationQuat : wrong number of arguments: %d, was expecting %d", argc, 0);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_setHeadRotationQuat(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    bool ok = true;
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_setHeadRotationQuat : Invalid Native Object");
    if (argc == 1) {
        cocos2d::Quaternion arg0;
        ok &= jsval_to_quaternion(cx, args.get(0), &arg0);
        JSB_PRECONDITION2(ok, cx, false, "js_cocos2dx_hmd_HMDScene_setHeadRotationQuat : Error processing arguments");
        cobj->setHeadRotationQuat(arg0);
        args.rval().setUndefined();
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_setHeadRotationQuat : wrong number of arguments: %d, was expecting %d", argc, 1);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_getCameraMask(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_getCameraMask : Invalid Native Object");
    if (argc == 0) {
        unsigned short ret = cobj->getCameraMask();
        jsval jsret = JSVAL_NULL;
        jsret = ushort_to_jsval(cx, ret);
        args.rval().set(jsret);
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_getCameraMask : wrong number of arguments: %d, was expecting %d", argc, 0);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters : Invalid Native Object");
    if (argc == 0) {
        double ret = cobj->getEyeDistanceMillimeters();
        jsval jsret = JSVAL_NULL;
        jsret = DOUBLE_TO_JSVAL(ret);
        args.rval().set(jsret);
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters : wrong number of arguments: %d, was expecting %d", argc, 0);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_getHeadPosition3D(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_getHeadPosition3D : Invalid Native Object");
    if (argc == 0) {
        cocos2d::Vec3 ret = cobj->getHeadPosition3D();
        jsval jsret = JSVAL_NULL;
        jsret = vector3_to_jsval(cx, ret);
        args.rval().set(jsret);
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_getHeadPosition3D : wrong number of arguments: %d, was expecting %d", argc, 0);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    bool ok = true;
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters : Invalid Native Object");
    if (argc == 1) {
        double arg0;
        ok &= JS::ToNumber( cx, args.get(0), &arg0) && !isnan(arg0);
        JSB_PRECONDITION2(ok, cx, false, "js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters : Error processing arguments");
        cobj->setEyeDistanceMillimeters(arg0);
        args.rval().setUndefined();
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters : wrong number of arguments: %d, was expecting %d", argc, 1);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_setHeadPosition3D(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    bool ok = true;
    JS::RootedObject obj(cx, args.thisv().toObjectOrNull());
    js_proxy_t *proxy = jsb_get_js_proxy(obj);
    cocos2d::HMDScene* cobj = (cocos2d::HMDScene *)(proxy ? proxy->ptr : NULL);
    JSB_PRECONDITION2( cobj, cx, false, "js_cocos2dx_hmd_HMDScene_setHeadPosition3D : Invalid Native Object");
    if (argc == 1) {
        cocos2d::Vec3 arg0;
        ok &= jsval_to_vector3(cx, args.get(0), &arg0);
        JSB_PRECONDITION2(ok, cx, false, "js_cocos2dx_hmd_HMDScene_setHeadPosition3D : Error processing arguments");
        cobj->setHeadPosition3D(arg0);
        args.rval().setUndefined();
        return true;
    }

    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_setHeadPosition3D : wrong number of arguments: %d, was expecting %d", argc, 1);
    return false;
}
bool js_cocos2dx_hmd_HMDScene_create(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    if (argc == 0) {
        cocos2d::HMDScene* ret = cocos2d::HMDScene::create();
        jsval jsret = JSVAL_NULL;
        do {
        if (ret) {
            js_proxy_t *jsProxy = js_get_or_create_proxy<cocos2d::HMDScene>(cx, (cocos2d::HMDScene*)ret);
            jsret = OBJECT_TO_JSVAL(jsProxy->obj);
        } else {
            jsret = JSVAL_NULL;
        }
    } while (0);
        args.rval().set(jsret);
        return true;
    }
    JS_ReportError(cx, "js_cocos2dx_hmd_HMDScene_create : wrong number of arguments");
    return false;
}

bool js_cocos2dx_hmd_HMDScene_constructor(JSContext *cx, uint32_t argc, jsval *vp)
{
    JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
    bool ok = true;
    cocos2d::HMDScene* cobj = new (std::nothrow) cocos2d::HMDScene();
    cocos2d::Ref *_ccobj = dynamic_cast<cocos2d::Ref *>(cobj);
    if (_ccobj) {
        _ccobj->autorelease();
    }
    TypeTest<cocos2d::HMDScene> t;
    js_type_class_t *typeClass = nullptr;
    std::string typeName = t.s_name();
    auto typeMapIter = _js_global_type_map.find(typeName);
    CCASSERT(typeMapIter != _js_global_type_map.end(), "Can't find the class type!");
    typeClass = typeMapIter->second;
    CCASSERT(typeClass, "The value is null.");
    // JSObject *obj = JS_NewObject(cx, typeClass->jsclass, typeClass->proto, typeClass->parentProto);
    JS::RootedObject proto(cx, typeClass->proto.get());
    JS::RootedObject parent(cx, typeClass->parentProto.get());
    JS::RootedObject obj(cx, JS_NewObject(cx, typeClass->jsclass, proto, parent));
    args.rval().set(OBJECT_TO_JSVAL(obj));
    // link the native object with the javascript object
    js_proxy_t* p = jsb_new_proxy(cobj, obj);
    AddNamedObjectRoot(cx, &p->obj, "cocos2d::HMDScene");
    if (JS_HasProperty(cx, obj, "_ctor", &ok) && ok)
        ScriptingCore::getInstance()->executeFunctionWithOwner(OBJECT_TO_JSVAL(obj), "_ctor", args);
    return true;
}


extern JSObject *jsb_cocos2d_Scene_prototype;

void js_cocos2d_HMDScene_finalize(JSFreeOp *fop, JSObject *obj) {
    CCLOGINFO("jsbindings: finalizing JS object %p (HMDScene)", obj);
}

void js_register_cocos2dx_hmd_HMDScene(JSContext *cx, JS::HandleObject global) {
    jsb_cocos2d_HMDScene_class = (JSClass *)calloc(1, sizeof(JSClass));
    jsb_cocos2d_HMDScene_class->name = "HMDScene";
    jsb_cocos2d_HMDScene_class->addProperty = JS_PropertyStub;
    jsb_cocos2d_HMDScene_class->delProperty = JS_DeletePropertyStub;
    jsb_cocos2d_HMDScene_class->getProperty = JS_PropertyStub;
    jsb_cocos2d_HMDScene_class->setProperty = JS_StrictPropertyStub;
    jsb_cocos2d_HMDScene_class->enumerate = JS_EnumerateStub;
    jsb_cocos2d_HMDScene_class->resolve = JS_ResolveStub;
    jsb_cocos2d_HMDScene_class->convert = JS_ConvertStub;
    jsb_cocos2d_HMDScene_class->finalize = js_cocos2d_HMDScene_finalize;
    jsb_cocos2d_HMDScene_class->flags = JSCLASS_HAS_RESERVED_SLOTS(2);

    static JSPropertySpec properties[] = {
        JS_PSG("__nativeObj", js_is_native_obj, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_PS_END
    };

    static JSFunctionSpec funcs[] = {
        JS_FN("addNode", js_cocos2dx_hmd_HMDScene_addNode, 1, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("getHeadRotationQuat", js_cocos2dx_hmd_HMDScene_getHeadRotationQuat, 0, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("setHeadRotationQuat", js_cocos2dx_hmd_HMDScene_setHeadRotationQuat, 1, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("getCameraMask", js_cocos2dx_hmd_HMDScene_getCameraMask, 0, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("getEyeDistanceMillimeters", js_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters, 0, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("getHeadPosition3D", js_cocos2dx_hmd_HMDScene_getHeadPosition3D, 0, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("setEyeDistanceMillimeters", js_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters, 1, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FN("setHeadPosition3D", js_cocos2dx_hmd_HMDScene_setHeadPosition3D, 1, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FS_END
    };

    static JSFunctionSpec st_funcs[] = {
        JS_FN("create", js_cocos2dx_hmd_HMDScene_create, 0, JSPROP_PERMANENT | JSPROP_ENUMERATE),
        JS_FS_END
    };

    jsb_cocos2d_HMDScene_prototype = JS_InitClass(
        cx, global,
        JS::RootedObject(cx, jsb_cocos2d_Scene_prototype),
        jsb_cocos2d_HMDScene_class,
        js_cocos2dx_hmd_HMDScene_constructor, 0, // constructor
        properties,
        funcs,
        NULL, // no static properties
        st_funcs);
    // make the class enumerable in the registered namespace
//  bool found;
//FIXME: Removed in Firefox v27 
//  JS_SetPropertyAttributes(cx, global, "HMDScene", JSPROP_ENUMERATE | JSPROP_READONLY, &found);

    // add the proto and JSClass to the type->js info hash table
    TypeTest<cocos2d::HMDScene> t;
    js_type_class_t *p;
    std::string typeName = t.s_name();
    if (_js_global_type_map.find(typeName) == _js_global_type_map.end())
    {
        p = (js_type_class_t *)malloc(sizeof(js_type_class_t));
        p->jsclass = jsb_cocos2d_HMDScene_class;
        p->proto = jsb_cocos2d_HMDScene_prototype;
        p->parentProto = jsb_cocos2d_Scene_prototype;
        _js_global_type_map.insert(std::make_pair(typeName, p));
    }
}

void register_all_cocos2dx_hmd(JSContext* cx, JS::HandleObject obj) {
    // Get the ns
    JS::RootedObject ns(cx);
    get_or_create_js_obj(cx, obj, "cc", &ns);

    js_register_cocos2dx_hmd_HMDScene(cx, ns);
}

