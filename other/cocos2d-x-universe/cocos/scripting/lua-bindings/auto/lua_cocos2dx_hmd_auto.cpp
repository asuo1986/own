#include "lua_cocos2dx_hmd_auto.hpp"
#include "CocosHMD.h"
#include "tolua_fix.h"
#include "LuaBasicConversions.h"


int lua_cocos2dx_hmd_HMDScene_addNode(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_addNode'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 1) 
    {
        cocos2d::Node* arg0;

        ok &= luaval_to_object<cocos2d::Node>(tolua_S, 2, "cc.Node",&arg0, "cc.HMDScene:addNode");
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_addNode'", nullptr);
            return 0;
        }
        cobj->addNode(arg0);
        lua_settop(tolua_S, 1);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:addNode",argc, 1);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_addNode'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_getCameraMask(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_getCameraMask'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 0) 
    {
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_getCameraMask'", nullptr);
            return 0;
        }
        unsigned short ret = cobj->getCameraMask();
        tolua_pushnumber(tolua_S,(lua_Number)ret);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:getCameraMask",argc, 0);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_getCameraMask'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 0) 
    {
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters'", nullptr);
            return 0;
        }
        double ret = cobj->getEyeDistanceMillimeters();
        tolua_pushnumber(tolua_S,(lua_Number)ret);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:getEyeDistanceMillimeters",argc, 0);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_getHeadPosition3D(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_getHeadPosition3D'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 0) 
    {
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_getHeadPosition3D'", nullptr);
            return 0;
        }
        cocos2d::Vec3 ret = cobj->getHeadPosition3D();
        vec3_to_luaval(tolua_S, ret);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:getHeadPosition3D",argc, 0);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_getHeadPosition3D'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 1) 
    {
        double arg0;

        ok &= luaval_to_number(tolua_S, 2,&arg0, "cc.HMDScene:setEyeDistanceMillimeters");
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters'", nullptr);
            return 0;
        }
        cobj->setEyeDistanceMillimeters(arg0);
        lua_settop(tolua_S, 1);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:setEyeDistanceMillimeters",argc, 1);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_setHeadPosition3D(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif


#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertype(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    cobj = (cocos2d::HMDScene*)tolua_tousertype(tolua_S,1,0);

#if COCOS2D_DEBUG >= 1
    if (!cobj) 
    {
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_setHeadPosition3D'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 1) 
    {
        cocos2d::Vec3 arg0;

        ok &= luaval_to_vec3(tolua_S, 2, &arg0, "cc.HMDScene:setHeadPosition3D");
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_setHeadPosition3D'", nullptr);
            return 0;
        }
        cobj->setHeadPosition3D(arg0);
        lua_settop(tolua_S, 1);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:setHeadPosition3D",argc, 1);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_setHeadPosition3D'.",&tolua_err);
#endif

    return 0;
}
int lua_cocos2dx_hmd_HMDScene_create(lua_State* tolua_S)
{
    int argc = 0;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif

#if COCOS2D_DEBUG >= 1
    if (!tolua_isusertable(tolua_S,1,"cc.HMDScene",0,&tolua_err)) goto tolua_lerror;
#endif

    argc = lua_gettop(tolua_S) - 1;

    if (argc == 0)
    {
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_create'", nullptr);
            return 0;
        }
        cocos2d::HMDScene* ret = cocos2d::HMDScene::create();
        object_to_luaval<cocos2d::HMDScene>(tolua_S, "cc.HMDScene",(cocos2d::HMDScene*)ret);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d\n ", "cc.HMDScene:create",argc, 0);
    return 0;
#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_create'.",&tolua_err);
#endif
    return 0;
}
int lua_cocos2dx_hmd_HMDScene_constructor(lua_State* tolua_S)
{
    int argc = 0;
    cocos2d::HMDScene* cobj = nullptr;
    bool ok  = true;

#if COCOS2D_DEBUG >= 1
    tolua_Error tolua_err;
#endif



    argc = lua_gettop(tolua_S)-1;
    if (argc == 0) 
    {
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_constructor'", nullptr);
            return 0;
        }
        cobj = new cocos2d::HMDScene();
        cobj->autorelease();
        int ID =  (int)cobj->_ID ;
        int* luaID =  &cobj->_luaID ;
        toluafix_pushusertype_ccobject(tolua_S, ID, luaID, (void*)cobj,"cc.HMDScene");
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:HMDScene",argc, 0);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_constructor'.",&tolua_err);
#endif

    return 0;
}

static int lua_cocos2dx_hmd_HMDScene_finalize(lua_State* tolua_S)
{
    printf("luabindings: finalizing LUA object (HMDScene)");
    return 0;
}

int lua_register_cocos2dx_hmd_HMDScene(lua_State* tolua_S)
{
    tolua_usertype(tolua_S,"cc.HMDScene");
    tolua_cclass(tolua_S,"HMDScene","cc.HMDScene","cc.Scene",nullptr);

    tolua_beginmodule(tolua_S,"HMDScene");
        tolua_function(tolua_S,"new",lua_cocos2dx_hmd_HMDScene_constructor);
        tolua_function(tolua_S,"addNode",lua_cocos2dx_hmd_HMDScene_addNode);
        tolua_function(tolua_S,"getCameraMask",lua_cocos2dx_hmd_HMDScene_getCameraMask);
        tolua_function(tolua_S,"getEyeDistanceMillimeters",lua_cocos2dx_hmd_HMDScene_getEyeDistanceMillimeters);
        tolua_function(tolua_S,"getHeadPosition3D",lua_cocos2dx_hmd_HMDScene_getHeadPosition3D);
        tolua_function(tolua_S,"setEyeDistanceMillimeters",lua_cocos2dx_hmd_HMDScene_setEyeDistanceMillimeters);
        tolua_function(tolua_S,"setHeadPosition3D",lua_cocos2dx_hmd_HMDScene_setHeadPosition3D);
        tolua_function(tolua_S,"create", lua_cocos2dx_hmd_HMDScene_create);
    tolua_endmodule(tolua_S);
    std::string typeName = typeid(cocos2d::HMDScene).name();
    g_luaType[typeName] = "cc.HMDScene";
    g_typeCast["HMDScene"] = "cc.HMDScene";
    return 1;
}
TOLUA_API int register_all_cocos2dx_hmd(lua_State* tolua_S)
{
	tolua_open(tolua_S);
	
	tolua_module(tolua_S,"cc",0);
	tolua_beginmodule(tolua_S,"cc");

	lua_register_cocos2dx_hmd_HMDScene(tolua_S);

	tolua_endmodule(tolua_S);
	return 1;
}

