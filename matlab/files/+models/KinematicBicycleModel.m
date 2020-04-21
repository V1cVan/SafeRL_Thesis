classdef KinematicBicycleModel < hwsim.models.Model
    %KBM Kinematic bicycle model
    
    methods
        function ds = derivatives(M,s,u)
            %DERIVATIVES Get the state derivatives for the given state and
            %inputs.
            %   * The state consists of the position (x,y and z) of the
            %     vehicle's CG, the angles (yaw,pitch and roll), velocity
            %     (longitudinal, lateral and vertical) of the vehicle's CG
            %     and angular velocities (yaw,pitch and roll rate).
            %   * The inputs consist of a longitudinal acceleration and
            %     steering angle.
            acc = u(1);% Note that this is a longitudinal acceleration. To get the lateral, we multiply by tan(beta)
            delta = u(2);
            % Apply bounds:
            acc = min(max(acc,M.LongAccBounds(1)),M.LongAccBounds(2));
            delta = min(max(delta,M.DeltaBounds(1)),M.DeltaBounds(2));
            % Calculate slip angle (beta) and total velocity
            beta = atan2(M.Lr*tan(delta),M.Lf+M.Lr);
            v = sqrt(s(7)*s(7)+s(8)*s(8));
            % Calculate derivatives of states:
            dx = v*cos(s(4)+beta);
            dy = v*sin(s(4)+beta);
            dpsi = v*sin(beta)/M.Lr;
            dvlong = acc;
            dvlat = acc*tan(delta)*M.Lr/(M.Lf+M.Lr);% Equal to acc*tan(beta)
            ds = [dx;dy;nan;dpsi;nan;nan;dvlong;dvlat;nan;nan;nan;nan];% Nan values are derivatives that are not provided by this model
        end
        
        function u_nom = nominal_inputs(M,~,roadInfo)
            %NOMINAL_INPUTS Get the inputs required for nominal control for
            %the given (augmented) state.
        	delta_nom = atan((M.Lf+M.Lr)/M.Lr*tan(-roadInfo.gamma));
            u_nom = [0;delta_nom];% Nominal acceleration equals zero
        end
    end
end

