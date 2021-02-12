classdef PID_controller < matlab.System
    % PID_controller Simple PID-controller to transform high-level steering actions into model inputs
    %   * The state consists of the previous error and the integral error
    %     accumulated so far.
    %   * The inputs consist of a longitudinal velocity or lateral offset error.
    %   * The props consist of the values for the proportional, derivative
    %     and integral gain K_p, K_d and K_i.

    % Public, tunable properties
    properties
        K_p = 0;
        K_i = 0;
        K_d = 0;
    end

    properties(DiscreteState)
        e_p % Previous error
        e_i % Integrated error
    end
    
    methods
        
        function ctrl = PID_controller(props)
            %PID_controller constructor
            ctrl@matlab.System();
            if nargin>0
                % Set model properties
                if isfield(props,'K_p')
                    ctrl.K_p = props.K_p;
                end
                if isfield(props,'K_i')
                    ctrl.K_i = props.K_i;
                end
                if isfield(props,'K_d')
                    ctrl.K_d = props.K_d;
                end
            end
        end
        
    end

    methods(Access = protected)
        
        function setupImpl(model)
            % Perform one-time calculations, such as computing constants
            model.e_p = 0;
            model.e_i = 0;
        end

        function y=stepImpl(model,dt,e)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            model.e_i = model.e_i + (e+model.e_p)*dt/2;
            y = model.K_p*e + model.K_i*model.e_i + model.K_d*(e-model.e_p)/dt;
            model.e_p = e;
        end

        function resetImpl(model)
            % Initialize / reset discrete-state properties
            model.e_p = 0;
            model.e_i = 0;
        end
    end
end

