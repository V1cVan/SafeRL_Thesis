classdef Vehicle < handle
    %VEHICLE Class representing any vehicle on the highway.
    
    properties (SetAccess = protected, GetAccess = public)
        model % mathematical model of the vehicle's movement in the global coordinate system
        driver % the driving policy that is being used to make the steering decisions for this vehicle
        col_status % collision status
        scenario % scenario the vehicle is driving in
        roadInfo % struct containing the augmented state information of the vehicle w.r.t. the road
    end
    
    properties (SetAccess = protected, GetAccess = public)
        state   % Current augmented state of the vehicle
        actions % Current actions as dictated by the vehicle's driving policy
    end
    
    properties
        pos % global position (x,y and z) of the vehicle's CG
        vel % longitudinal, lateral and vertical velocity of the vehicle
        ang % Yaw, pitch and roll of the vehicle in the global coordinate system
        ang_vel % Yaw, pitch and roll rate of the vehicle
        % Link to a controller, accepting high-level steering commands
        % (reference longitudinal velocity and lateral offset) and
        % returning suitable low-level steering inputs for the vehicle's
        % dynamical model
        vel_ctrl
        lat_ctrl
        % Reference to a plotted patch
        p
    end
    
    properties (Constant)
        % Modifiable constants
        D_max = 50 % Detect all vehicles within range D_max (in meters)
        N_max = 10 % Only use state information of at most N_max vehicles within range
        
        % Fixed constants
        COL_NONE = 0 % No collision status
        COL_LEFT = -1 % Collision with left road boundary status
        COL_RIGHT = -2 % Collision with right road boundary status
    end
    
    methods
        function V = Vehicle(props,model,driver,scenario)
            %VEHICLE Construct an instance of this class
            V.model = model;
            V.driver = driver;
            V.scenario = scenario;
            if isempty(scenario)
                % Dummy vehicle, used for plotting
                V.pos = props.pos;
                V.vel = props.vel;
                V.ang = props.ang;
            else
                % Simulated vehicle
                V.roadInfo = struct("id",props.road,"pos",props.pos);
                [x,y,yaw] = scenario.roads(props.road).getGlobalPose(props.pos(1),props.pos(2),props.heading);
                V.pos = [x;y;0];
                V.vel = props.vel;
                V.ang = [yaw;0;0];
                V.ang_vel = [0;0;0];

                % Manually tuned PID controllers for longitudinal and lateral
                % actions:
                import('controllers.*');
                V.vel_ctrl = PID_controller(struct('K_p',3.5)); % Proportional velocity controller
%                 V.lat_ctrl = PID_controller(struct('K_p',0.08,'K_i',0.001,'K_d',0.05)); % PID controller for lateral action
                V.lat_ctrl = PID_controller(struct('K_p',0.08)); % PID controller for lateral action
                % TODO: lateral controller for low velocities?
                %V.lat_ctrl = Stanley_controller(struct('ct',1)); % Not properly working with the used parameters
                V.col_status = V.COL_NONE;

                V.updateRoadInfo();
            end
        end
        
        function updateDriverState(V,otherInfo,distances,offsets)
            % The augmented state vector consists of:
            %   1:2 The vehicle's lateral offset w.r.t. the right and left
            %       boundary (accounting for the vehicle's road dimensions)
            %   3   The vehicle's lateral offset w.r.t. the nearest lane
            %       center.
            %   4:5 The vehicle's lateral offset w.r.t. the nearest lane
            %       center to the right and left (equal to state(3) if
            %       there is no lane to the right/left)
            %   6   The difference between the maximum allowed speed and
            %       the vehicle's longitudinal velocity
            %   7:8 The vehicle's velocity in both longitudinal and lateral
            %       direction of the road
            %   9:4:end The concatenated states of other surrounding vehicles that are
            %   within a distance d_max of the EV (or capped at N_max vehicles).
            %   Their state is encoded w.r.t. the EV as follows:
            %       The relative lateral offset (positive if EV is to the
            %       left, negative if it is to the right)
            %       The estimated relative longitudinal offset, calculated
            %       as sqrt(d^2-(d_lat_EV-d_lat)^2) and positive if EV is
            %       in front, negative if it is behind.
            %       The relative velocity in both longitudinal and lateral
            %       direction of the road (i.e. speed corrigated for difference
            %       between the vehicle's yaw and lane heading) (positive if EV's
            %       velocity is faster, negative otherwise)
            V.state = zeros(8+4*V.N_max,1);
            V.col_status = V.COL_NONE;
            % First fill the state vector with the vehicle's own state
            V.state(1:2) = V.roadInfo.boundaryOffset;
            V.state(3) = V.roadInfo.centerOffset;
            V.state(4:5) = V.roadInfo.neighbourOffset;
            V.state(6) = V.roadInfo.maxVel-V.roadInfo.roadVel(1);
            V.state(7:8) = V.roadInfo.roadVel;
            if V.roadInfo.boundaryOffset(1)<0
                V.col_status = V.COL_RIGHT;
            end
            if V.roadInfo.boundaryOffset(2)<0
                V.col_status = V.COL_LEFT;
            end
            for ov=1:numel(distances) % Then for each other nearby vehicle: append its relative state to the state vector
                dlat_cg = offsets(2,ov);
                V.state(5+ov*4) = max(0,abs(dlat_cg)-(otherInfo(ov).roadSize(2)+V.roadInfo.roadSize(2))/2);% Account for both vehicle's lateral dimensions along the road
                V.state(5+ov*4) = V.roadInfo.direction*sign(dlat_cg)*V.state(5+ov*4);% Determine correct sign of lateral offset
                dlong = max(0,sqrt(max(0,distances(ov)^2-dlat_cg^2))-(otherInfo(ov).roadSize(1)+V.roadInfo.roadSize(1))/2);% Account for both vehicle's lognitudinal dimensions along the road
                V.state(6+ov*4) = V.roadInfo.direction*sign(offsets(1,ov))*dlong;% Determine sign of longitudinal offset
                V.state(7+ov*4:8+ov*4) = V.roadInfo.roadVel-otherInfo(ov).roadVel;% Relative velocities
                if V.state(5+ov*4)==0 && V.state(6+ov*4)==0
                    % Collision when lateral and longitudinal offset w.r.t. ov is zero
                    V.col_status = ov;
                end
            end
            for ov=numel(distances)+1:V.N_max
                % Append with dummy relative states:
                V.state([5:8]+4*ov) = [0;V.D_max;0;0];
            end
            V.actions = V.driver.drive(V.state); % Get driving actions from the driver policy based on the current vehicle's state
        end
        
        function inputs = advanceTime(V,dt)
            % actions   The actions returned by the vehicle's driver
        	% inputs    The inputs to the vehicle's dynamical model
            inputs = V.model.nominal_inputs([V.pos;V.ang;V.vel;V.ang_vel],V.roadInfo);% Get nominal control inputs
            % TODO: shouldn't the controllers get the previous dt?
            inputs(1) = inputs(1)+V.vel_ctrl(dt,V.actions(1)-V.vel(1)); % Get acceleration input from longitudinal velocity error
            inputs(2) = inputs(2)+V.lat_ctrl(dt,V.actions(2)); % Get steering angle input from lateral offset error
            % Update state of the model based on the current actions and
            % resulting inputs:
            next_state = utils.integrateRK4(@(rs) V.getRoadDerivatives(rs,inputs),[V.roadInfo.pos;V.pos;V.ang;V.vel;V.ang_vel],dt);
            [V.roadInfo.id,V.roadInfo.pos(1),V.roadInfo.pos(2)] = V.updateRoadState(next_state(1),next_state(2));
            [V.pos(1),V.pos(2)] = V.scenario.roads(V.roadInfo.id).getGlobalPose(V.roadInfo.pos(1),V.roadInfo.pos(2),0);
            %V.pos(1:2) = next_state(3:4);% Gets inaccurate for large simulation times
            V.ang(1) = next_state(6);
            V.vel(1:2) = next_state(9:10);
            V.ang_vel(1) = next_state(12);
            % Update roadInfo:
            V.updateRoadInfo();
        end
        
        function plot(V,ax,fc,ec)
            D = [V.model.Lf,V.model.Lr;
                 V.model.Size(2)/2,V.model.Size(2)/2;
                 V.model.Hcg,V.model.Size(3)-V.model.Hcg];
            [Vertices,Faces] = utils.getCuboidPatch(V.pos,D,V.ang(1));
            if ishghandle(V.p)
                % Update existing patch
                V.p.Vertices = Vertices;
                if ~isempty(fc)
                    V.p.FaceColor = fc;
                end
                if ~isempty(ec)
                    V.p.EdgeColor = ec;
                end
            else
                % Create new patch
                V.p = patch(ax,'Vertices',Vertices,'Faces',Faces,'FaceColor',fc,'EdgeColor',ec);
            end
        end
        
        function s = saveobj(V)
            s.model = V.model;
            s.driver = V.driver;
            s.props = struct("pos",V.pos,"vel",V.vel,"ang",V.ang);
            % Do not save scenario, roadInfo, controllers and p
        end
        
    end
    
    methods(Static)
        function V = loadobj(s)
            if isstruct(s)
                V = Vehicle(s.props,s.model,s.driver,[]);
            else
                V = s;
            end
       end
    end
    
    methods(Access=private)
        function updateRoadInfo(V)
            %UPDATEROADINFO Calculates an augmented state representation
            %for this vehicle based on its position on the road.
            % The augmented road state consists of:
            %   5:6     The vehicle's dimensions projected along the direction of the road
            %   7:8     The vehicle's lateral offset w.r.t. both road boundaries
            %           (this lateral offset is w.r.t. the vehicle's closest point)
            %   9       The vehicle's lateral offset w.r.t. the nearest lane center
            %           (this lateral offset is w.r.t. the vehicle's CG)
            %   10:11   The vehicle's velocity in both longitudinal and lateral
            %           direction (of the road)
            
            % R,s and l are already correctly set by the advanceTime method
            R = V.roadInfo.id;
            s = V.roadInfo.pos(1);
            l = V.roadInfo.pos(2);
            
            % With the updated road coordinates, get the road layout at the vehicle's current position:
            [lb,ln] = V.scenario.roads(R).getBoundaries(s);
            [le,lc,lw] = V.scenario.roads(R).getLaneEdges(s,1:size(lb,2));
            L = V.scenario.roads(R).getLaneId(s,l);% Current lane id
            G = Road.getContiguousGroup(lb,ln,L);
            dir = V.scenario.roads(R).getDirection(L);
            
            boundaryOffset = dir*[l-le(1,G(1),1);le(1,G(end),2)-l];% W.r.t. right and left boundary
            laneWidth = lw(L);
            centerOffset = dir*(l-lc(L));
            la = Road.getAvailability(lb,L);
            neighbourOffset = [centerOffset;centerOffset];% W.r.t. right and left neighbouring lane
            neighbourOffset(la) = dir*(l-lc(ln(1,L,la)));
            gamma = V.ang(1)-V.scenario.roads(R).heading(s,l);% NOTE: this incorporates the angle offset of the lane w.r.t. the road outline's heading!!
            height = V.scenario.roads(R).evalProp('height',s,L);
            
            % Calculate effective lateral and longitudinal dimensions
            % (along the road) of the vehicle
            RM = [cos(abs(gamma)),sin(abs(gamma));sin(abs(gamma)),cos(abs(gamma))];
            roadSize = RM*V.model.Size(1:2); % longitudinal and lateral dimensions (along the road) of the vehicle
            
            % Map longitudinal and lateral velocities of the vehicle to
            % longitudinal and lateral 'lane' velocities by using the headingAngle
            RM = [cos(gamma),sin(gamma);-sin(gamma),cos(gamma)];
            roadVel = RM*V.vel(1:2);
            
            % Update roadInfo:
            V.roadInfo.lane = L;
            V.roadInfo.gamma = gamma;
            V.roadInfo.laneWidth = laneWidth;
            V.roadInfo.direction = dir;
            V.roadInfo.roadSize = roadSize;
            V.roadInfo.maxVel = V.scenario.roads(R).evalProp('speed',s,L);
            V.roadInfo.roadVel = roadVel;
            V.roadInfo.boundaryOffset = boundaryOffset-roadSize(2)/2;
            V.roadInfo.centerOffset = centerOffset;
            V.roadInfo.neighbourOffset = neighbourOffset;
            
            % Set z coordinate, as it is not considered by the vehicle's
            % dynamical model:
            V.pos(3) = height+V.model.Hcg;
        end
        
        function dxa = getRoadDerivatives(V,xa,inputs)
            % Get the derivatives of the road dynamical system from the
            % augmented state (road position,global state)
            [real_R,real_s,real_l,d] = V.updateRoadState(xa(1),xa(2));
            x = xa(3:end);
            dx = V.model.derivatives(x,inputs);% Calculate global dynamical system derivatives
            psi_r = V.scenario.roads(real_R).heading(real_s,[]);
            kappa_r = V.scenario.roads(real_R).curvature(real_s,0);
            RM = [cos(psi_r),sin(psi_r);-sin(psi_r),cos(psi_r)];
            dxa = [d*RM*dx(1:2);dx];
            dxa(1) = dxa(1)/(1-real_l*kappa_r);
        end
        
        function [real_R,real_s,real_l,d] = updateRoadState(V,s,l)
            % Get a new valid road state from the given integrated road
            % coordinates s and l (who are not wrapped in case of lane
            % connections). d denotes a change in direction in going from
            % one lane to another. It is 1 if the direction stays the same
            % and -1 if the direction flips.
            old_R = V.roadInfo.id;
            old_s = V.roadInfo.pos(1);
            old_l = V.roadInfo.pos(2);
            delta_s = s-old_s;
            delta_l = l-old_l;
            [real_R,real_s,real_l,d] = V.scenario.updateRoadState(old_R,old_s,old_l,delta_s,delta_l);
        end
    end
end

