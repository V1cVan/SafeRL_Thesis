classdef utils
    %UTILS Utility functions
    
    methods(Static)
        function xn = integrateRK4(sys,x,dt)
            % Integrate the given system from the given state for a time
            % step dt.
            k1 = dt*sys(x);
            k2 = dt*sys(x+k1/2);
            k3 = dt*sys(x+k2/2);
            k4 = dt*sys(x+k3);
            xn = x+(k1+2*k2+2*k3+k4)/6;
        end
        
        function [V,F] = getCuboidPatch(C,S,yaw)
            % Get the vertices and faces matrices of a cuboid positioned
            % around C of dimensions S (first column denoting the front,
            % left and lower part of the cuboid w.r.t. C ; second column
            % denoting the rear, right and upper part of the cuboid w.r.t.
            % C) and rotated by yaw.
            R = [cos(yaw),sin(yaw);-sin(yaw),cos(yaw)];
            V = C(1:2)' + [-S(1,2),-S(2,2);
                            S(1,1),-S(2,2);
                            S(1,1), S(2,1);
                           -S(1,2), S(2,1)]*R;% Vertices of 2D 'ground face'
            if size(S,1)>2
                V = [V,ones(4,1)*(C(3)-S(3,1));V,ones(4,1)*(C(3)+S(3,2))];
                F = [1,2,3,4;
                     1,2,6,5;
                     2,3,7,6;
                     3,4,8,7;
                     4,1,5,8;
                     5,6,7,8];
            else
                F = [1,2,3,4];
            end
        end
    end
end

