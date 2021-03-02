classdef ManualDriver < hwsim.drivers.Driver
    %MANUALDRIVER Class representing a basic manual driving policy
    
    properties (GetAccess=public,SetAccess=protected)
        state;
        des_vel_diff; % Desired velocity of this driver w.r.t. the maximum allowed speed (in m/s)
        overtake_gap = 15; % when gap between successive vehicles becomes smaller than this value, we will try to overtake
        type = hwsim.drivers.ManualDriver.DEFENSIVE;
        color;
    end
    
    properties (GetAccess=protected,SetAccess=protected)
        overtaking = false;
        % lane changes take ~6s
    end
    
    properties (Constant,GetAccess=public)
        SLOW = 1;
        DEFENSIVE = 2;
        AGGRESSIVE = 3;
    end
    
    properties (Constant,GetAccess=public)
        SAFETY_GAP = 5; % minimum gap between vehicles we want to ensure (in meters)
        TTC = 5; % minimum TTC we want to ensure (in seconds)
        EPS = 1/20; % Small offset that is allowed
        FRONT_VEL_SCALE = 1.05; % Scale factor. All vehicles with speed smaller than FRONT_VEL_SCALE*D.des_vel will be taken into account.
    end
    
    methods
        function D = ManualDriver(props)
            %MANUALDRIVER Construct an instance of this class.
            D@hwsim.drivers.Driver();
            if nargin > 0 && isfield(props,'driver_type')
                D.type = props.driver_type;
            end
            min_vel = 1; max_vel = 1;
            if D.type==D.SLOW
                min_vel = -5;
                max_vel = -2;
                D.overtake_gap=0; % Slow vehicles will never overtake
                D.color = [0.494,0.184,0.556];% Purple
            elseif D.type==D.DEFENSIVE
                min_vel = -2;
                max_vel = 1;
                D.overtake_gap = 15;
                D.color = [0,0.447,0.741];% Dark blue
            elseif D.type==D.AGGRESSIVE
                min_vel=1;
                max_vel=4;
                D.overtake_gap=0.8*hwsim.Vehicle.D_max; % Aggressive vehicles will try to overtake very fast
                D.color = [0.85,0.325,0.098];% Orange
            end
            D.des_vel_diff = rand()*(max_vel-min_vel)+min_vel;
            D.state = struct("front_d",nan,"front_v",nan,"left_free",nan,"right_free",nan);
        end
        
        function actions = drive(D,state)
            %DRIVE Execute one step of the driving policy based on the
            %current state of the vehicle.
            dlat_boundary = state(1:2);
            dlat_center = state(3);
            dlat_neighb = state(4:5);
            des_vel = state(6)+state(7)+D.des_vel_diff;
            v_long = state(7);
            v_lat = state(8);
            actions = [des_vel,-dlat_center]; % Default action is desired speed and corrigate to center of lane
            
            N_ov = (numel(state)-8)/4;% Number of detected other vehicles
            % We calculate a simplified vehicle state from the provided
            % state vector. This simplified state only takes the closest
            % vehicle in front of us into account (and only if it is not
            % moving much faster than us) together with two flags
            % indicating whether we can move one lane to the left or right.
            % Note that the current lane is defined as the lane in which
            % the vehicle's center of gravity lies (i.e. the lane for which
            % the center offset is calculated).
            front_d = hwsim.Vehicle.D_max; front_v = D.FRONT_VEL_SCALE*des_vel; % Default is: car in front far ahead and at our desired speed
            lw = abs(dlat_neighb-dlat_center);% Estimates of the lane width of the right and left lane
            right_free = lw(1)>D.EPS;% Default is: we can move right if there is a lane available to the right
            left_free = lw(2)>D.EPS;% Default is: we can move left if there is a lane available to the left
            % We are not in the leftmost/rightmost lane when dlat_neighb is
            % not equal to dlat_center.
            for i=1:N_ov
                os = state(5+i*4:8+i*4);
                LR = 1+(1-sign(os(1)))/2;% 1 if OV is to the right, 2 if it is to the left of EV
                if os(1)==0
                    % There is some lateral overlap between us and the
                    % other vehicle
                    if os(2)<0 && -os(2)<front_d && v_long-os(3)<D.FRONT_VEL_SCALE*des_vel
                        % If there is a vehicle directly in front of us that is
                        % going slower than our desired speed (up to a margin) and
                        % is closer than the current front_d, update the front_d
                        % and front_v to this vehicle's state values.
                        front_d = -os(2);
                        front_v = v_long-os(3);
                        % I.e. we only care about the closest vehicle in
                        % front of us that is also going slower than us
                    end
                    % The other cases correspond to a collision (which
                    % should stop the simulation) and a vehicle behind us
                    % (which we do not care about)
                elseif abs(os(1)+dlat_center)<lw(LR)-D.EPS
                    % There is no lateral overlap between us and the other
                    % vehicle, but there is also no full lane between us
                    
                    % Otherwise the vehicle is far enough away
                    % (in lateral offset), such that the left and right
                    % lane remain free.
                    if os(2)==0
                        % There is longitudinal overlap between us and the
                        % other vehicle
                        left_free = left_free && (os(1)>0); % Then the left is still free if the other vehicle is to the right
                        right_free = right_free && (os(1)<0); % And the right is still free if the other vehicle is to the left
                    elseif os(2)<0% && os(3)>0
                        % There is a vehicle in front that is going slower
                        % than us
                        if os(1)<0
                            % And it is to the left
                            left_free = left_free && (-os(2)>os(3)*D.TTC) && (-os(2)>D.SAFETY_GAP);
                            % Then the left is still free if we won't catch
                            % up on it within 5 seconds and there is enough
                            % distance to impose the safety gap
                        else % os(1)>0
                            % And it is to the right
                            right_free = right_free && (-os(2)>os(3)*D.TTC) && (-os(2)>D.SAFETY_GAP) && (-os(2)>D.overtake_gap);
                            % Then the right is still free if we won't catch
                            % up on it within 5 seconds and there is enough
                            % distance to impose the safety and overtake gap
                        end
                    elseif os(2)>0% && os(3)<0
                        % There is a vehicle behind us that is going faster
                        % than us
                        if os(1)<0
                            % And it is to the left
                            left_free = left_free && (os(2)>-os(3)*D.TTC) && (os(2)>D.SAFETY_GAP);
                            % Then the left is still free if it won't catch
                            % up on us within 5 seconds and there is enough
                            % distance to impose the safety gap
                        else % os(1)>0
                            % And it is to the right
                            right_free = right_free && (os(2)>-os(3)*D.TTC) && (os(2)>D.SAFETY_GAP);
                            % Then the right is still free if it won't catch
                            % up on us within 5 seconds and there is enough
                            % distance to impose the safety gap
                        end
                    end
                    % The other cases correspond to a vehicle in front that
                    % is going faster than us and a vehicle behind us that
                    % is going slower than us. In both cases both the left
                    % and right remain free for us.
                end
                % The other cases correspond to a vehicle that is more than
                % 1 lane width away from us in lateral offset.
            end
            
            % Update state variable for outside reference:
            D.state.front_d = front_d;
            D.state.front_v = front_v;
            D.state.left_free = left_free;
            D.state.right_free = right_free;
            
            % Now decide which actions we will take based on our driving
            % type:
            if front_d < hwsim.Vehicle.D_max
                % If there is a vehicle in front of us, linearly
                % adapt speed to match front_v.
                alpha = (front_d-D.SAFETY_GAP)/(hwsim.Vehicle.D_max-D.SAFETY_GAP);
                actions(1) = min(des_vel,(1-alpha)*front_v+alpha*des_vel);
            end
            should_overtake = (front_d < D.overtake_gap && front_v < 0.9*des_vel);
            if should_overtake && ~D.overtaking
                % If we are getting close to a vehicle in front of
                % us that is going slower than our desired speed,
                % try to overtake it
                D.overtaking = true;
            end
            if D.overtaking
                if (abs(dlat_center) < D.EPS && ~should_overtake) || (~left_free && dlat_center > -D.EPS)
                    % If we are done overtaking 1 lane and we
                    % shouldn't continue to overtake yet another
                    % lane, set overtaking to false.
                    % OR if the left is no longer free while we are
                    % still on the previous lane, we should return
                    % to the lane center and stop overtaking.
                    D.overtaking = false;
                elseif left_free && dlat_center > -D.EPS
                    % If the left lane is free and we are still on
                    % the previous lane, go left
                    actions(2) = -dlat_neighb(2);
                end
                % In the other case we are already on the
                % next lane so we should first wait to get to the
                % middle of the lane before deciding to overtake to
                % yet another lane.
            elseif right_free % Otherwise, if we can go to the right, go there
                actions(2) = -dlat_neighb(1);
            end
        end
    end
end

