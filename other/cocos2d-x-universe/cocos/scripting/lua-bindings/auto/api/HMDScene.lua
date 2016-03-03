
--------------------------------
-- @module HMDScene
-- @extend Scene
-- @parent_module cc

--------------------------------
--  Add child to be rendered in the stereo view<br>
-- param node Node to add.<br>
-- js NA
-- @function [parent=#HMDScene] addNode 
-- @param self
-- @param #cc.Node node
-- @return HMDScene#HMDScene self (return value: cc.HMDScene)
        
--------------------------------
--  get the full camera mask. This is required to render in both "eyes"<br>
-- js NA
-- @function [parent=#HMDScene] getCameraMask 
-- @param self
-- @return unsigned short#unsigned short ret (return value: unsigned short)
        
--------------------------------
-- 
-- @function [parent=#HMDScene] getEyeDistanceMillimeters 
-- @param self
-- @return float#float ret (return value: float)
        
--------------------------------
-- 
-- @function [parent=#HMDScene] getHeadPosition3D 
-- @param self
-- @return vec3_table#vec3_table ret (return value: vec3_table)
        
--------------------------------
--  Set the distance between the left and right "eyes"<br>
-- param dist The distance in millimeters<br>
-- js NA
-- @function [parent=#HMDScene] setEyeDistanceMillimeters 
-- @param self
-- @param #float dist
-- @return HMDScene#HMDScene self (return value: cc.HMDScene)
        
--------------------------------
--  set the position of the head (eyes will update)<br>
-- param pos Position in world space.<br>
-- js NA
-- @function [parent=#HMDScene] setHeadPosition3D 
-- @param self
-- @param #vec3_table pos
-- @return HMDScene#HMDScene self (return value: cc.HMDScene)
        
--------------------------------
-- 
-- @function [parent=#HMDScene] create 
-- @param self
-- @return HMDScene#HMDScene ret (return value: cc.HMDScene)
        
--------------------------------
-- 
-- @function [parent=#HMDScene] HMDScene 
-- @param self
-- @return HMDScene#HMDScene self (return value: cc.HMDScene)
        
return nil
