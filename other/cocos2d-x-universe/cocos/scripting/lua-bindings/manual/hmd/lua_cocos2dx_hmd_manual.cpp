#include "lua_cocos2dx_hmd_manual.hpp"
#include "lua_cocos2dx_3d_auto.hpp"
#include "CocosHMD.h"
#include "tolua_fix.h"
#include "LuaBasicConversions.h"
#include "CCLuaEngine.h"


using namespace cocos2d;

int lua_cocos2dx_hmd_HMDScene_setHeadRotationQuat(lua_State* tolua_S)
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
        tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_setHeadRotationQuat'", nullptr);
        return 0;
    }
#endif

    argc = lua_gettop(tolua_S)-1;
    if (argc == 1) 
    {
        cocos2d::Quaternion arg0;
      
        ok &= luaval_to_quaternion(tolua_S, 2,&arg0);
        if(!ok)
        {
            tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_setHeadRotationQuat'", nullptr);
            return 0;
        }
        cobj->setHeadRotationQuat(arg0);
        lua_settop(tolua_S, 1);
        return 1;
    }
    luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:setHeadRotationQuat",argc, 1);
    return 0;

#if COCOS2D_DEBUG >= 1
    tolua_lerror:
    tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_setHeadRotationQuat'.",&tolua_err);
#endif

    return 0;
}

int lua_cocos2dx_hmd_HMDScene_getHeadRotationQuat(lua_State* tolua_S)
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
    tolua_error(tolua_S,"invalid 'cobj' in function 'lua_cocos2dx_hmd_HMDScene_getHeadRotationQuat'", nullptr);
    return 0;
  }
#endif
  
  argc = lua_gettop(tolua_S)-1;
  if (argc == 0)
  {
    if(!ok)
    {
      tolua_error(tolua_S,"invalid arguments in function 'lua_cocos2dx_hmd_HMDScene_getHeadRotationQuat'", nullptr);
      return 0;
    }
    cocos2d::Quaternion ret = cobj->getHeadRotationQuat();
    quaternion_to_luaval(tolua_S,(cocos2d::Quaternion)ret);
    return 1;
  }
  luaL_error(tolua_S, "%s has wrong number of arguments: %d, was expecting %d \n", "cc.HMDScene:getHeadRotationQuat",argc, 0);
  return 0;
  
#if COCOS2D_DEBUG >= 1
tolua_lerror:
  tolua_error(tolua_S,"#ferror in function 'lua_cocos2dx_hmd_HMDScene_getHeadRotationQuat'.",&tolua_err);
#endif
  
  return 0;
}



int extendHMDScene(lua_State* L)
{
    lua_pushstring(L, "cc.HMDScene");
    lua_rawget(L, LUA_REGISTRYINDEX);
    if (lua_istable(L,-1))
    {
      tolua_function(L, "setHeadRotationQuat", lua_cocos2dx_hmd_HMDScene_setHeadRotationQuat);
      tolua_function(L, "getHeadRotationQuat", lua_cocos2dx_hmd_HMDScene_getHeadRotationQuat);
    }
    lua_pop(L, 1);
    return 1;
}

TOLUA_API int register_all_cocos2dx_hmd_manual(lua_State* tolua_S)
{
  extendHMDScene(tolua_S);
	return 1;
}

