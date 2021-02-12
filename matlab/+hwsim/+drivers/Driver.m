classdef (Abstract) Driver < handle
    %DRIVER Abstract base class of a driving policy
    
    methods (Abstract)
        %DRIVE Execute one step of the driving policy based on the
        %current state of the vehicle.
        actions = drive(state)
    end
end

