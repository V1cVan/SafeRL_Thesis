classdef Switching_controller < matlab.System
    % Switching_controller Switching controller, changing the controller based on the current regime of the system
    %   * The inputs consist of a reference input and actual input
    %   * The props consist of the different controllers to use with
    %     specifications about which regime they apply to

    % Public, tunable properties
    properties
        controllers
        regimes
    end

    properties(DiscreteState)
    end
    
    methods
        
        function ctrl = Switching_controller(controllers,regimes)
            %PID_controller constructor
            ctrl@matlab.System();
            % Set sample time and other model properties
            ctrl.controllers = controllers;
            ctrl.regimes = regimes;
        end
        
    end

    methods(Access = protected)
        
        function setupImpl(ctrl)
            % Perform one-time calculations, such as computing constants
            %for i=1:numel(ctrl.controllers)
            %    setup(ctrl.controllers{i});
            %end
        end

        function y=stepImpl(ctrl,dt,e,v)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            for i=1:numel(ctrl.controllers)
                yi = ctrl.controllers{i}(e,dt);
                if isfield(ctrl.regimes{i},'e_range') && isfield(ctrl.regimes{i},'v_range')
                    if ctrl.regimes{i}.e_range(1)<=abs(e) && ctrl.regimes{i}.e_range(2)>=abs(e) && ctrl.regimes{i}.v_range(1)<=abs(v) && ctrl.regimes{i}.v_range(2)>=abs(v)
                        y = yi;
                    end
                elseif isfield(ctrl.regimes{i},'e_range')
                    if ctrl.regimes{i}.e_range(1)<=abs(e) && ctrl.regimes{i}.e_range(2)>=abs(e)
                        y = yi;
                    end
                elseif isfield(ctrl.regimes{i},'v_range')
                    if ctrl.regimes{i}.v_range(1)<=abs(v) && ctrl.regimes{i}.v_range(2)>=abs(v)
                        y = yi;
                    end
                else
                    y = yi;
                end
            end
        end

        function resetImpl(ctrl)
            % Initialize / reset discrete-state properties
            for i=1:numel(ctrl.controllers)
                reset(ctrl.controllers{i});
            end
        end
    end
end

