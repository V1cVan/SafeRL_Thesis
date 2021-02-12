classdef Stanley_controller < matlab.System
    % Stanley_controller Used to transform desired lateral offset into a steering angle action
    %   * The state consists of the previous error and the integral error
    %     accumulated so far.
    %   * The inputs consist of the lateral position error and current vehicle pose and longitudinal velocity.
    %   * The props consist of the distance of the values for the
    %     proportional and integral gain K_p and K_i.

    % Public, tunable properties
    properties
        ct = 1; % Completion time of a steering maneuver in seconds per meter (of lateral offset to compensate)
    end

    properties(DiscreteState)
        % No discrete state
    end
    
    methods
        
        function ctrl = Stanley_controller(props)
            %Stanley_controller constructor
            ctrl@matlab.System();
            % Set sample time and other model properties
            if isfield(props,'ct')
                ctrl.ct = props.ct;
            end
        end
        
    end

    methods(Access = protected)
        
        function setupImpl(model)
            % Perform one-time calculations, such as computing constants
        end

        function y=stepImpl(model,currPose,currVel,hAngle,eLat)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            refPose = zeros(1,3);
            t = model.ct*abs(eLat);% Estimated completion time of the maneuver
            R = [cos(currPose(3)),sin(currPose(3));-sin(currPose(3)),cos(currPose(3))];
            refPose(1:2) = currPose(1:2)'+[cos(currPose(3));-sin(currPose(3))]*currVel(1)*t+eLat*[sin(currPose(3)+hAngle);-cos(currPose(3)+hAngle)];
            % refPose = currPose + distance travelled by longitudinal speed
            % + lateral error
            refPose(3) = rad2deg(currPose(3)+hAngle);% Reference yaw should be aligned with yaw of the road
            currPose(3) = rad2deg(currPose(3));
            y = deg2rad(lateralControllerStanley(refPose,currPose,currVel(1)));
        end

        function resetImpl(model)
            % Initialize / reset discrete-state properties
        end
    end
end

