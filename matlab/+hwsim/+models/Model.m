classdef (Abstract) Model < handle
    %MODEL Abstract base class of a vehicle dynamics model
    
    properties
        LongAccBounds   % Bounds for the longitudinal acceleration
        DeltaBounds     % Bounds for the steering angle
        Size            % Size of the vehicle (longitudinal, lateral and height)
        Lf              % Distance of the front wheels w.r.t. the CG
        Lr              % Distance of the rear wheels w.r.t. the CG
        Hcg             % Height of CG w.r.t. the wheel axles
    end
    
    methods
        function M = Model(props)
            %MODEL Construct an instance of this class
            arguments
                props.LongAccBounds (2,1) double = [-5;5]
                props.DeltaBounds (2,1) double = [-0.1;0.1]
                props.Size (3,1) double
                props.RelCG (2,1) double
            end
            M.LongAccBounds = props.LongAccBounds;
            M.DeltaBounds = props.DeltaBounds;
            M.Size = props.Size;
            M.Lf = props.RelCG(1)*M.Size(1);
            M.Lr = (1-props.RelCG(1))*M.Size(1);
            M.Hcg = props.RelCG(2)*M.Size(3);
        end
    end
    
    methods (Abstract)
        %DERIVATIVES Get the state derivatives for the given state and
        %inputs.
        dx = derivatives(M,x,u)
        %NOMINAL_INPUTS Get the inputs required for nominal control for the
        %given state and roadInfo.
        u_nom = nominal_inputs(M,x,roadInfo)
    end
end

