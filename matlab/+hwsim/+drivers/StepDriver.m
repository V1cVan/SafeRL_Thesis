classdef StepDriver < hwsim.drivers.Driver
    %STEPDRIVER Class representing a stepping driving policy, used to
    %examine the step response of the dynamical systems.
    
    properties
        color = [0.635,0.078,0.184];
    end
    
    properties (GetAccess=protected,SetAccess=protected)
        vel_range = [0,40];
        switch_steps = 10*10;
        switch_timer = 0;
        cur_vel;
        cur_dlat;
    end
    
    methods
        function D = StepDriver(props)
            %BASICDRIVER Construct an instance of this class.
            D@hwsim.drivers.Driver();
            if nargin>0
                if isfield(props,'min_vel')
                    D.vel_range(1) = props.min_vel;
                end
                if isfield(props,'max_vel')
                    D.vel_range(2) = props.max_vel;
                end
                if isfield(props,'switch_steps')
                    D.switch_steps = props.switch_steps;
                end
            end
            D.switch_timer = D.switch_steps; % Enforce new setpoint on next drive action
        end
        
        function actions = drive(D,state)
            %DRIVE Execute one step of the driving policy based on the
            %current state of the vehicle.
            D.switch_timer = D.switch_timer + 1;
            dlat_right = state(1);
            dlat_left = state(2);
            if D.switch_timer >= D.switch_steps
                D.switch_timer = 0;
                D.cur_vel = (D.vel_range(2)-D.vel_range(1))*rand()+D.vel_range(1);
                D.cur_dlat = rand()*0.8+0.1;
            end
            actions = [D.cur_vel,D.cur_dlat*(dlat_left+dlat_right)-dlat_right];
        end
    end
end

