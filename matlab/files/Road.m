classdef Road < handle
    %ROAD Class representing a single road as a piecewise clothoid curve
    % TODO: include connection points (from a lane on this road to a lane
    % on another road or from a lane on this road that is being removed to
    % another lane on this road), this can be used to calculate smoothed
    % road centers (and their derivatives) in transition zones.
    
    properties
        outline
        lane_props
        
        SPL
        s
        theta
        kappa
        length
        
        cyclic
        useMEX = false;
    end
    
    properties(Access=private)
        mexHandle = []
    end
    
    properties(Constant,Access=private)
        EPS = 10^-6;% Account for small floating point errors
    end
    
    methods
        function R = Road(data,useMEX)
            %Road Construct an instance of this class
            
            % Save original road specifications
            R.outline = data.outline;
            R.cyclic = isempty(data.bc);
            R.lane_props = data.lanes;
            
            % Calculate the piecewise clothoid specification
            ROOT = fileparts(mfilename('fullpath'));% Path to folder containing this source file
            addpath(fullfile(ROOT,'../../Clothoids/matlab'));% TODO: remove this dependency
            S = ClothoidSplineG2();
            S.verbose(false);
            if R.cyclic
                R.outline(end+1,:) = R.outline(1,:);
                R.SPL = S.buildP2(R.outline(:,1),R.outline(:,2));% Cyclic G2 interpolation
            elseif any(isnan(data.bc))
                R.SPL = S.buildP4(R.outline(:,1),R.outline(:,2));% Minimal curvature G2 interpolation
            else
                R.SPL = S.buildP1(R.outline(:,1),R.outline(:,2),data.bc(1),data.bc(2));% G2 interpolation with start and end slope given
            end
            [R.s,R.theta,R.kappa] = R.SPL.getSTK();
            R.length = R.s(end);
            R.s = R.s(1:end-1);% Required for Road.getSegment
            
            % Process some layout properties:
            for lane=1:numel(R.lane_props)
                f = R.lane_props(lane).validity{1};
                t = R.lane_props(lane).validity{2};
                if isempty(f)
                    f = 0;
                end
                if isempty(t)
                    t = R.length();
                end
                R.lane_props(lane).validity = [f,t];
            end
            
            % Initialize C++ object:
            if nargin>1
                R.useMEX = useMEX;
            end
            if R.useMEX
                if isempty(R.lane_props) % Empty lane props are given by the scenario designer to create a connection road
                    R.lane_props = struct([]);
                end
                R.mexHandle = mexRoad('new',data.cp,R.lane_props);
            end
        end
        
        function delete(R)
            % Destroy C++ object:
            if ~isempty(R.mexHandle)
                mexRoad('delete',R.mexHandle);
            end
            R.mexHandle = [];
        end
        
        function [p,dp] = evalProp(R,prop,s,lanes)
            % Calculate the lane property (e.g. lateral offset or lane width)
            % and its derivative for the given lanes at the given curvilinear abscissa s.
            if nargin<4
                lanes = 1:numel(R.lane_props);
            end
            
            if R.useMEX
                [p,dp] = mexRoad(prop,R.mexHandle,s,lanes);
            else
                p = nan(numel(s),numel(lanes));
                dp = nan(numel(s),numel(lanes));

                for i=1:numel(lanes)
                    L = lanes(i);
                    v_range = s>=R.lane_props(L).validity(1) & s<=R.lane_props(L).validity(2);
                    [p(v_range,i),dp(v_range,i)] = Property.evaluate(R.lane_props(L).(prop),s(v_range));
                end
            end
        end
        
        function I = getSegment(R,s)
            I = find(R.s<=s,1,'last');
        end
        
        function psi = heading(R,s,l)
            if R.useMEX
                psi = mexRoad('heading',R.mexHandle,s,l);
            else
                [~,~,psi] = R.SPL.evaluate(s);
                if ~isempty(l)
                    L = R.getLaneId(s,l);
                    [~,dlc] = R.evalProp('offset',s);
                    dlc = dlc(sub2ind(size(dlc),1:size(dlc,1),L'));
                    dir = [R.lane_props.direction];
                    psi = psi+atan2(dlc,(1-l'.*R.curvature(s,0)))+(1-dir(L))*pi/2;
                end
            end
        end
        
        function kappa = curvature(R,s,l)
            if R.useMEX
                kappa = mexRoad('curvature',R.mexHandle,s,l);
            else
                kappa = R.SPL.kappa(s);
                kappa = kappa./(1-l.*kappa);
            end
        end
        
        function [x,y,yaw] = getGlobalPose(R,s,l,gamma)
            % Calculate the global position (x,y) and yaw angle of a point
            % on this road based on its road position (s,l) and gamma
            % angle.
            [x,y,psi_r] = R.SPL.evaluate(s);% Get pose of road outline
            x = x-l*sin(psi_r);
            y = y+l*cos(psi_r);
            psi_l = R.heading(s,l);
            yaw = psi_l+gamma;
        end
        
        function L = getLaneId(R,s,l)
            if R.useMEX
                L = mexRoad('laneId',R.mexHandle,s,l);
            else
                lc = R.evalProp('offset',s);
                D = abs(lc-l);
                M = max(max(D));
                D(isnan(D)) = M;
                [~,L] = min(D,[],2);
            end
        end
        
%         function lm = getActiveLaneMask(R,s)
%             val = vertcat(R.lane_props.validity)';
%             lm = false(numel(s),numel(R.lane_props));
%             for L=1:numel(R.lane_props)
%                 C = s>=val(1,L) & s<=val(2,L);% Indices within validity range of L
%                 lm(:,L) = C;
%             end
%         end
        
        function [le,lc,lw] = getLaneEdges(R,s,lanes)
            lc = R.evalProp('offset',s,lanes);
            lw = R.evalProp('width',s,lanes);
            dir = [R.lane_props(lanes).direction];
            le = nan([size(lc),2]);
            le(:,:,1) = lc-dir.*lw/2;
            le(:,:,2) = lc+dir.*lw/2;
        end
        
        function ln = getLaneNeighbours(R,s,lanes)
            % Returns for each entry in s the lane id of the first valid
            % lane directly to the right and left of each lane.
            if R.useMEX
                ln = mexRoad('neighbours',R.mexHandle,s,lanes);
            else
                ln = nan(numel(s),numel(lanes),2);
                for i=1:numel(lanes)
                    L = lanes(i);
                    ln(:,i,1) = R.getLaneNeighbour(s,L,-1);
                    ln(:,i,2) = R.getLaneNeighbour(s,L,1);
                end
            end
        end
        
        function [lb,ln] = getBoundaries(R,s)
            % Returns the lane boundary types for the given lanes and given
            % values of s. The boundary type is 0 for an edge that cannot
            % be crossed, 1 for an edge that can only be crossed from the
            % respective lane, 2 for an edge that can only be crossed from
            % the respective lane's neighbour and 3 for an edge that can be
            % crossed from both the respective lane and its neighbour. If
            % the lane is not valid for a given s, the boundary type is NaN
            if R.useMEX
                [lb,ln] = mexRoad('boundaries',R.mexHandle,s,1:numel(R.lane_props));
            else
                ld = [R.lane_props.direction];
                lh = R.evalProp('height',s);
                le = R.getLaneEdges(s,1:numel(R.lane_props));
                ln = R.getLaneNeighbours(s,1:numel(R.lane_props));
                la = nan(numel(s),numel(R.lane_props),2);
                la(:,:,1) = R.evalProp('right',s);
                la(:,:,2) = R.evalProp('left',s);

                lb = zeros(numel(s),numel(R.lane_props),2);
                S = (1:numel(s))';% Used for indexing below

                for L=1:numel(R.lane_props)
                    lb(isnan(le(:,L,1)),L,:) = nan;% NaN boundary type where the lane is invalid
                    for d=[1,2]
                        e = R.lane_props(L).direction*d+(1-R.lane_props(L).direction)*3/2;% Flips d for lanes in opposite direction
                        I = ~isnan(ln(:,L,e));% Indices for which we have a neighbour
                        N = ln(I,L,e);% Lane id of our neighbour(s)
                        if ~isempty(N)
                            A = abs(lh(I,L)-lh(sub2ind(size(lh),S(I),N)))<Road.EPS;% Mask for I denoting for which indices the neighbour is at the same height,
                            A = A & (ld(L)==ld(N))';% has the same direction and
                            A = A & (3-2*d)*(le(I,L,e)-le(sub2ind(size(le),S(I),N,(3-e)*ones(nnz(I),1))))<Road.EPS;% for which indices there is no gap between this lane and its neighbour
                            I(I) = A;% Apply availability mask
                            lb(I,L,e) = la(I,L,d)+2*la(sub2ind(size(la),S(I,1),ln(I,L,e),3-d*ones(nnz(I),1)));% +1 if we can cross towards the neighbour ; +2 if we can cross towards this lane from the neighbour
                        end
                    end
                end
            end
        end
        
        function [cF,cT] = getConnections(R,L)
            cF = cat(1,R.lane_props(L).from);
            cT = cat(1,R.lane_props(L).to);
        end
        
        function Lm = getTargetLane(R,L)
            Lm = R.lane_props(L).merge;
        end
        
        function lv = getValidity(R,lanes)
            lv = cat(1,R.lane_props(lanes).validity);
        end
        
        function ld = getDirection(R,lanes)
            ld = [R.lane_props(lanes).direction];
        end
        
        function plot(R,ax,s,opts)
            arguments
                R (1,1) Road
                ax = gca
                s (1,:) double = []
                opts.Lanes (1,:) char {mustBeMember(opts.Lanes,{'on','off'})} = 'on'
                opts.Flatten (1,:) char {mustBeMember(opts.Flatten,{'on','off'})} = 'off'
                opts.OutlineColor = []
                opts.OutlineMarkers (1,:) char {mustBeMember(opts.OutlineMarkers,{'on','off'})} = 'off'
                opts.BoundaryColor = 'k'
                opts.LaneCenters (1,:) char {mustBeMember(opts.LaneCenters,{'on','off'})} = 'off'
                opts.LaneHighlight = []
            end
            val = vertcat(R.lane_props.validity)';
            if isempty(s)
                %s=linspace(0,R.length,round(R.length));
                s = [];
                delta = 0.1;
                C = unique([0;val(:);R.length]);% Construct s such that all values at the start and end of the road's lanes are included
                s = C(1);
                p = C(1)+delta;
                for i=2:numel(C)
                    newRange = linspace(p,C(i)-delta,round(C(i)-p));
                    s = [s,newRange,C(i)];
                    p = C(i)+delta;
                end
            end
            % Evaluate clothoid curve and get road layout for the specified
            % interval of s
            [x,y,psi]=R.SPL.evaluate(s);
            [le,lc]=R.getLaneEdges(s,1:numel(R.lane_props));
            lh = R.evalProp('height',s);
            if strcmp(opts.Flatten,'on')
                lh = 0*lh;
            end
            if strcmp(opts.Lanes,'on')
                % Calculate all lane edge curves:
                xle = x'-le.*sin(psi');
                yle = y'+le.*cos(psi');
                % Calculate all lane boundary types:
                bs = R.getBoundarySpans(s);
                % Draw lane bodies:
                for L=1:numel(R.lane_props)
                    I = 1:numel(s);
                    I = I(s>=val(1,L) & s<=val(2,L));% Indices within validity range
                    patch(ax,[xle(I,L,1);xle(I(end:-1:1),L,2)],[yle(I,L,1);yle(I(end:-1:1),L,2)],[lh(I,L);lh(I(end:-1:1),L)],[0.7,0.7,0.7],'LineStyle','none'); % Draw lane body
                end
                % Draw lane edges:
                for L=1:numel(R.lane_props)
                    for d=[1,2]
                        % Right edge is 1, left edge is 2
                        for i=1:size(bs{L,d},1)
                            type = bs{L,d}(i,1);
                            span = bs{L,d}(i,2):bs{L,d}(i,3);
                            % 0:    Full line
                            % 1:    Dashed line, in combination with a full line
                            % 2:    Full line, in combination with a dashed line
                            % 3:    Shared dashed line
                            switch(type)
                                case 0
                                    % Full line
                                    line(ax,xle(span,L,d),yle(span,L,d),lh(span,L),"Color",opts.BoundaryColor,"LineWidth",2);
                                case 1
                                    % Dashed line, in combination with a full line
                                    off=-(2*d-3)/20;
                                    th=R.heading(s(span),le(span,L));
                                    line(ax,xle(span,L,d)-off*sin(th'),yle(span,L,d)+off*cos(th'),lh(span,L),"Color","w","LineStyle","--","LineWidth",1);
                                    line(ax,xle(span,L,d),yle(span,L,d),lh(span,L),"Color","w","LineWidth",1);
                                case 2
                                    % Full line, in combination with a dashed line
                                    off=-(2*d-3)/20;
                                    th=R.heading(s(span),le(span,L));
                                    line(ax,xle(span,L,d)-off*sin(th'),yle(span,L,d)+off*cos(th'),lh(span,L),"Color","w","LineWidth",1);
                                    line(ax,xle(span,L,d),yle(span,L,d),lh(span,L),"Color","w","LineStyle","--","LineWidth",1);
                                case 3
                                    % Shared dashed line
                                    line(ax,xle(span,L,d),yle(span,L,d),lh(span,L),"Color","w","LineStyle","--","LineWidth",1);
                            end
                        end
                    end
                end
            end
            
            if ~isempty(opts.OutlineColor)
                line(ax,x,y,'Color',opts.OutlineColor,'LineWidth',5);
                if strcmp(opts.OutlineMarkers,'on')
                    line(ax,R.outline(:,1),R.outline(:,2),'Color',opts.OutlineColor,'LineStyle','none','LineWidth',5,'Marker','d');
                end
            end
            if strcmp(opts.LaneCenters,'on')
                for i=1:size(lc,2)
                    line(ax,x'-lc(:,i).*sin(psi'),y'+lc(:,i).*cos(psi'),lh(:,i),'Color','b');
                end
            end
            for L=opts.LaneHighlight
                I = 1:numel(s);
                I = I(s>=val(1,L) & s<=val(2,L));% Indices within validity range
                line(ax,xle(I,L,1),yle(I,L,1),lh(I,L),'Color',[0.85,0.325,0.098,0.75],'LineWidth',2);
                line(ax,xle(I,L,2),yle(I,L,2),lh(I,L),'Color',[0.85,0.325,0.098,0.75],'LineWidth',2);
            end
        end
        
        function Analyse(R)
            figure(3);
            R.SPL.plot(); % Plot clothoid curve
            xlabel('x (m)');
            ylabel('y (m)');
            figure(4);
            R.SPL.plotCurvature(400); % Plot curvature
            xlabel('s (m)');
            ylabel('\kappa (rad/m)');
        end
    end
    
    % Public methods used by the scenario designer and multi_car_simulation
    methods
        function [po,s,I] = getClosestPoint(R,pi)
            % Returns the point p0 on the clothoid curve that is closest to
            % the given point pi. Also returns the s value of that point
            % and the index I of the segment p0 belongs to.
            [x,y,s] = R.SPL.closestPoint(pi(1),pi(2));
            po = [x,y];
            I = R.getSegment(s);
        end
        
        function [ML,Ms] = getLaneMapping(R)
            lv = R.getValidity(1:numel(R.lane_props));
            ML = nan(numel(R.lane_props)+1,5);
            Ms = nan(2*numel(R.lane_props)+1,5);
            ML(1,:) = [0,0,0,0,1];
            ps = 0;
            for L=1:numel(R.lane_props)
                range = lv(L,:);
                if isempty(R.lane_props(L).from) && R.lane_props(L).width(1,5)==0
                    % Quite an arbitrary check to see if the lane is being
                    % inserted. In this case, only allow vehicles to be
                    % placed on this lane after it is fully inserted.
                    % TODO: very bad way of handling this, we cannot be
                    % sure the width property perfectly follows the made
                    % assumptions here
                    d = 1+(1-R.lane_props(L).direction)/2;
                    I = find(R.lane_props(L).width(:,1)==range(d));
                    [~,J] = max(R.lane_props(L).direction*R.lane_props(L).width(I,2));
                    range(d) = R.lane_props(L).width(I(J),2);
                end
                D = range(2)-range(1);
                ML(L+1,:) = [ML(L,1:2)+D,0,0,1];% Step to next lane number
                Ms(2*L-1,:) = [ML(L,1),ML(L,1),0,0,range(1)-ps];% Step to valid from s-value
                Ms(2*L,:) = [ML(L,1),ML(L,1)+D,1,0,D];% Linearly go to valid to s-value
                ps = range(2);
            end
            ML(end,5) = -numel(R.lane_props);% Step back to zero
            Ms(end,:) = [ML(end,1),ML(end,1),0,0,-ps];% Step back to zero
        end
        
        function ns = getNeighbourSpans(R,L,LR)
            % Should remain like this, without taking lane direction into
            % account. Only used by scenario_designer's gap functionality.
            val = vertcat(R.lane_props.validity)';
            range = val(:,L);
            if LR>0
                lanes = L+1:size(val,2);
            else
                lanes = L-1:-1:1;
            end
            ns = [];
            while range(1)~=range(2)
                i = 1;
                while i<=numel(lanes) && (val(1,lanes(i))>range(1) || val(2,lanes(i))<=range(1))
                    % Find next valid neighbour
                    i = i+1;
                end
                if i>numel(lanes)
                    break;
                end
                N = lanes(i);
                ns(end+1,:) = [N,range(1),min(val(2,lanes(i)),range(2))];
                range(1) = min(val(2,lanes(i)),range(2));
            end
        end
    end
    
    methods(Static)
        function lg = getContiguousGroup(lb,ln,L)
            lg = L;
            while lb(1,lg(1),1)~=0
                lg = [ln(1,lg(1),1),lg];% Add right neighbours until we encounter an uncrossable right boundary
            end
            while lb(1,lg(end),2)~=0
                lg = [lg,ln(1,lg(end),2)];% Add left neighbours until we encounter an uncrossable left boundary
            end
        end
        
        function la = getAvailability(lb,lanes)
            la = (lb(:,lanes,:)==1) | (lb(:,lanes,:)==3);
        end
    end
    
    methods(Access=private)
        function bs = getBoundarySpans(R,s)
            % Returns all lane boundaries types of this road. 0 for a
            % boundary that cannot be crossed (full line), 1 for a boundary
            % that can only be crossed from the current lane, 2 for a
            % boundary that can only be crossed from the neighbouring lane,
            % 3 for a boundary that can be crossed in both directions.
            bs = cell(numel(R.lane_props),2);
            [lb,ln] = R.getBoundaries(s);
            
            for L=1:numel(R.lane_props)
                for d=[1,2] % right boundary = 1 ; left boundary = 2
                    % First calculate a boundary types vector for the whole span:
                    b = lb(:,L,d);
                    if ~isempty(R.lane_props(L).merge)
                        I = ~isnan(ln(:,L,d));% Indices for which we have a neighbour
                        A = R.lane_props(L).merge==ln(I,L,d);% Mask for those indices for which the neighbour equals the one we merge with
                        A = A & b(I)>0;% And we share a crossable boundary with this neighbour
                        I(I) = A;
                        b(I) = -1;% Put non existent type for these regions, as these crossable edges will be drawn by that neighbour
                    end
                    b(isnan(b)) = -1;% And also for the region where the lane is invalid

                    % Next, convert the boundary types vector into spans for each type:
                    c = abs(b(2:end)-b(1:end-1));% Get spans created by boundary type changes
                    ln(isnan(ln(:,L,d)),L,d) = -1;% Convert nans to a non existent lane number
                    c = c+abs(ln(2:end,L,d)-ln(1:end-1,L,d));% Get spans created by neighbour changes
                    c = [c;-1];% Make sure the last span is also added (see for loop below)
                    S = find(c);
                    s0 = 1;% Start point of first span
                    bs{L,d} = [];
                    for i=1:numel(S)
                        type = b(S(i));
                        if type>=0 % Skip non existent type
                            bs{L,d}(end+1,:) = [type,s0,S(i)];
                        end
                        s0 = S(i)+1;
                    end
                end
            end
        end
        
        function ln = getLaneNeighbour(R,s,L,LR)
            val = vertcat(R.lane_props.validity)';
            ln = nan(numel(s),1);
            if LR*R.lane_props(L).direction>0
                lanes = size(val,2):-1:L+1;
            else
                lanes = 1:L-1;
            end
            for lane=lanes
                I = 1:numel(s);
                C = s>=val(1,L) & s<=val(2,L);% Indices within validity range of L
                C = C & s>=val(1,lane) & s<=val(2,lane);% And within validity range of lane
                I = I(C);
                ln(I) = lane;
            end
        end
    end
end

