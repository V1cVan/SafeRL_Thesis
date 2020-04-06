classdef DynamicBicycleModel < models.Model
    %DBM Dynamic bicycle model
    
    properties
        m       % Mass of the vehicle
        Izz     % Moment of inertia about vehicle-fixed z-axis
        Fn      % Nominal normal force applied to axles, along vehicle-fixed z-axis
        w       % Front and rear track widths
        Cy      % Front and rear wheel cornering stiffness
        mu      % Front and rear wheel friction coefficient
    end
    
    properties(Constant)
        g = 9.81 % Gravitational acceleration
    end
    
    methods
        function M = DynamicBicycleModel(own_props,base_props)
            %KBM Construct an instance of this class
            arguments
                own_props.m (1,1) double
                own_props.Izz (1,1) double
                own_props.Fn (1,1) double
                own_props.w (1,1) double
                own_props.Cy (1,1) double
                own_props.mu (1,1) double
                base_props.?Model
            end
            base_props = namedargs2cell(base_props);
            M@models.Model(base_props{:});
            M.m = own_props.m;
            M.Izz = own_props.Izz;
            M.Fn = own_props.Fn;
            M.w = own_props.w;
            M.Cy = own_props.Cy;
            M.mu = own_props.mu;
        end
        
        function ds = derivatives(M,s,u)
            %DERIVATIVES Get the state derivatives for the given state and
            %inputs.
            %   * The state consists of the position (x and y) of the
            %     vehicle's CG, the yaw, velocity (both longitudinal and
            %     lateral) of the vehicle's CG and yaw rate.
            %   * The inputs consist of a longitudinal acceleration and
            %     steering angle.
            acc = u(1);% Note that this is a longitudinal acceleration. To get the lateral, we multiply by tan(beta)
            delta = u(2);
            % Apply bounds:
            acc = min(max(acc,M.LongAccBounds(1)),M.LongAccBounds(2));% TODO: move bound calculation to super class
            delta = min(max(delta,M.DeltaBounds(1)),M.DeltaBounds(2));
            ds = nan(size(s));% Not implemented
        end
        
        function u_nom = nominal_inputs(M,s,roadInfo)
            %NOMINAL_INPUTS Get the inputs required for nominal control for
            %the given (augmented) state.
        	delta_nom = nan;% Not implemented
            u_nom = [0;delta_nom];% Nominal acceleration equals zero
        end
    end
end

