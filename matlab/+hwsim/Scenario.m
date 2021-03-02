classdef Scenario
    %Scenario DrivingScenario defined by road centers.
    
    properties (SetAccess=protected, GetAccess=public)
        roads
    end
    
    methods
        function S = Scenario(scenario)
            %SCENARIO Construct an instance of this class
            if isstring(scenario) || ischar(scenario)
                % Load the scenario from the json data file
                scenarios = load('scenarios/scenarios.mat');
                scenario = scenarios.(scenario);
            end
            S.roads = hwsim.Road.empty();
            for r=1:numel(scenario.roads)
                S.roads(r) = hwsim.Road(scenario.roads(r));
            end
        end
        
        function [roadId,s,l,d] = updateRoadState(S,roadId,s,l,ds,dl)
            % wrap the given road state to a new valid road state, given
            % the updates ds and dl, using the lane connections defined by
            % the road layout. Note that (roadId,s,l) should be a VALID
            % starting state.
            L = S.roads(roadId).getLaneId(s,l);
            [~,cT] = S.roads(roadId).getConnections(L);
            vF = S.roads(roadId).getValidity(L);
            dirF = S.roads(roadId).getDirection(L);
            d = 1;
            if (s+ds<vF(1) || s+ds>vF(2))
                % We cross the end of the current lane
                if ~isempty(cT)
                    dl = l+dl-S.roads(roadId).evalProp('offset',vF(1+(1+dirF)/2),L);
                    roadId = cT(1);
                    L = cT(2);
                    vT = S.roads(roadId).getValidity(L);
                    dirT = S.roads(roadId).getDirection(L);
                    d = dirF*dirT;
                    s = vT(1+(1-dirT)/2)+d*(s+ds-vF(1+(1+dirF)/2));
                    l = S.roads(roadId).evalProp('offset',vT(1+(1-dirT)/2),L)+d*dl;
                elseif ~isempty(S.roads(roadId).getTargetLane(L))
                    s = s+ds;
                    l = l+dl;
                else
                    % No connection and no target lane => end of simulation for this vehicle
                    roadId = nan;
                    s = nan;
                    l = nan;
                end
            else
                s = s+ds;
                l = l+dl;
            end
        end
        
        function offsets = getRoadOffsets(S,V_ref,V_other)
            if ~isempty(V_other)
                R_ref = V_ref.roadInfo.id;
                pos_ref = V_ref.roadInfo.pos;
                dir_ref = V_ref.roadInfo.direction;
                otherInfo = [V_other.roadInfo];
                R = [otherInfo.id];
                pos = [otherInfo.pos];

                lanes = 1:numel(S.roads(R_ref).lane_props);
                val = S.roads(R_ref).getValidity(lanes);
                dir = S.roads(R_ref).getDirection(lanes)';
                Is = val(:,1+(1-dir_ref)/2)>=pos_ref(1)-hwsim.Vehicle.D_max & val(:,1+(1-dir_ref)/2)<=pos_ref(1)+hwsim.Vehicle.D_max & dir==dir_ref;% Lanes that start within the vehicle's detection horizon
                Ie = val(:,1+(1+dir_ref)/2)>=pos_ref(1)-hwsim.Vehicle.D_max & val(:,1+(1+dir_ref)/2)<=pos_ref(1)+hwsim.Vehicle.D_max & dir==dir_ref;% Lanes that end within the vehicle's detection horizon
                Ls = lanes(Is);
                Le = lanes(Ie);
                [cF,~] = S.roads(R_ref).getConnections(Ls);
                [~,cT] = S.roads(R_ref).getConnections(Le);

                offsets = pos_ref-pos;
                if ~isempty(cF)
                    R_F = unique(cF(:,1));
                    for i=1:numel(R_F)
                        I = cF(:,1)==R_F(i);
                        L_F = cF(I,2);% All lanes of road R_F(i) that have a connection towards
                        L_r = Ls(I);% these lanes of road R_ref

                        s_r = val(L_r(1),1+(1-dir_ref)/2);% s value of connection on road R_ref
                        dir_F = S.roads(R_F(i)).getDirection(L_F(1));
                        val_F = S.roads(R_F(i)).getValidity(L_F(1));
                        s_F = val_F(1+(1+dir_F)/2);% s value of connection on road R_F(i)
                        dl = sum(S.roads(R_ref).evalProp('offset',s_r,L_r))/numel(L_r);% Lateral shift introduced by the connection

                        % Only select vehicles on road R_F(i) that are within
                        % the detection horizon in the region BEFORE the
                        % connection (otherwise issues with cyclic connections)
                        I = R==R_F(i) & dir_F*pos(1,:)<=dir_F*s_F & dir_F*pos(1,:)>=dir_F*s_F-(hwsim.Vehicle.D_max-dir_ref*(pos_ref(1)-s_r));
                        offsets(1,I) = pos_ref(1)-(s_r+dir_F*dir_ref*(pos(1,I)-s_F));
                        offsets(2,I) = pos_ref(2)-(dl+dir_F*dir_ref*pos(2,I));
                    end
                end
                if ~isempty(cT)
                    R_T = unique(cT(:,1));
                    for i=1:numel(R_T)
                        I = cT(:,1)==R_T(i);
                        L_T = cT(I,2);% All lanes of road R_T(i) that have an incoming connection from
                        L_r = Le(I);% these lanes of road R_ref

                        s_r = val(L_r(1),1+(1+dir_ref)/2);% s value of connection on road R_ref
                        dir_T = S.roads(R_T(i)).getDirection(L_T(1));
                        val_T = S.roads(R_T(i)).getValidity(L_T(1));
                        s_T = val_T(1+(1-dir_T)/2);% s value of connection on road R_T(i)
                        dl = sum(S.roads(R_ref).evalProp('offset',s_r,L_r))/numel(L_r);% Lateral shift introduced by the connection

                        % Only select vehicles on road R_T(i) that are within
                        % the detection horizon in the region AFTER the
                        % connection (otherwise issues with cyclic connections)
                        I = R==R_T(i) & dir_T*pos(1,:)>=dir_T*s_T & dir_T*pos(1,:)<=dir_T*s_T+(hwsim.Vehicle.D_max+dir_ref*(pos_ref(1)-s_r));
                        offsets(1,I) = pos_ref(1)-(s_r+dir_T*dir_ref*(pos(1,I)-s_T));
                        offsets(2,I) = pos_ref(2)-(dl+dir_T*dir_ref*pos(2,I));
                    end
                end
            else
                offsets = [];
            end
        end
        
        function plot(S,varargin)
            for r=1:numel(S.roads)
                S.roads(r).plot(varargin{:});
            end
        end
        
        function Analyse(S)
            for r=1:numel(S.roads)
                figure(1);
                S.roads(r).plot('OutlineColor',[0.6,0.6,0.6]);
                S.roads(r).Analyse();
            end
        end
        
        function [MR,ML,Ms] = getRoadMapping(S)
            MR = nan(numel(S.roads)+1,5);
            ML = [];
            Ms = [];
            MR(1,:) = [0,0,0,0,1];
            import('hwsim.Property')
            for r=1:numel(S.roads)
                [MLr,Msr] = S.roads(r).getLaneMapping();
                ML = Property.combine(ML,Property.translate(MLr,MR(r,1)),1,1);
                Ms = Property.combine(Ms,Property.translate(Msr,MR(r,1)),1,1);
                length = MLr(end,1);
                MR(r+1,:) = [MR(r,1:2)+length,0,0,1];% Step to next road ID
            end
            MR(end,5) = -numel(S.roads);% Step back to zero
        end
    end
end

