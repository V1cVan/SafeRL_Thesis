classdef Property
    % Utility methods to easily perform calculations with road properties.
    
    methods(Static)
        function [p,dp] = evaluate(P,s)
            % Evaluate the value of a lane property for the given
            % curvilinear abscissa s. The property values are the sum of a
            % number of independent transitions. Each transition is
            % constant up to its start point s0 and also onwards from its
            % end point s1. In between there is a smooth interpolation of
            % the property values based on the type.
            % P encodes all transitions for this property as an Nx5 matrix
            % where each column defines (in order):
            %   * s0: the s values denoting the start of each transition
            %   * s1: the s values denoting the end of each transition
            %   * t: type of the transition (0 for cosine-smoothed and 1 for linear)
            %   * p0: property value at the start of each transition
            %   * p1: property value at the end of each transition
            p = zeros(numel(s),1);
            dp = zeros(numel(s),1);
            
            for i=1:size(P,1)
                % Get the indices of all s values before and after this
                % transition:
                before = s<=P(i,1);
                after = s>P(i,2);
                
                % Account for transitions in negative direction:
                if P(i,1)>P(i,2)
                    before = ~before;
                    after = ~after;
                end
                
                % Get the indices of all s values in between the
                % transition's start and end:
                between = ~before & ~after;
                
                % Update the property values for all values of s:
                p(before) = p(before) + P(i,4);
                u = (s(between)-P(i,1))/(P(i,2)-P(i,1));% Default is linear transition
                du = ones(size(u))/(P(i,2)-P(i,1));
                I = u<=0.5;
                if P(i,3)==0 % No smoothing (heaviside transition)
                    du = 0;
                    u(I) = 0;
                    u(~I) = 1;
                elseif P(i,3)==2 % Quadratic smoothing
                    du(I) = 4*u(I).*du(I);
                    du(~I) = 4*(1-u(~I)).*du(~I);
                    u(I) = 2*u(I).*u(I);
                    u(~I) = 1-2*(1-u(~I)).*(1-u(~I));
                elseif P(i,3)==3 % Cosine smoothing
                    du = pi*sin(u*pi)/2.*du;
                    u = 0.5-cos(u*pi)/2;
                end
                p(between) = p(between) + P(i,4)+u'*(P(i,5)-P(i,4));% Use an interpolated layout
                dp(between) = dp(between) + du'*(P(i,5)-P(i,4));
                p(after) = p(after) + P(i,5);
            end
        end
        
        function P = shift(P,dp)
            P(:,5) = P(:,5)+dp;
        end
        
        function P = translate(P,ds)
            P(:,1:2) = P(:,1:2)+ds;
        end
        
        function P = combine(P1,P2,a1,a2,s_res,p_res)
            % Calculates simplify(P,s_res,p_res) where 'P=a1*P1+a2*P2'
            if a1==0 || isempty(P1)
                P = [];
            else
                P = [P1(:,1:3),a1*P1(:,4:5)];
            end
            if a2~=0 && ~isempty(P2)
                P = [P;P2(:,1:3),a2*P2(:,4:5)];
            end
            if nargin>4
                P = Property.simplify(P,s_res,p_res);
            end
        end
        
        function P = simplify(P,s_res,p_res)
            if isempty(P)
                return;
            end
            i = 2;
            while i<=size(P,1)
                % Merge fully overlapping transitions:
                I = find((abs(P(1:i-1,1)-P(i,1))<s_res) & (abs(P(1:i-1,2)-P(i,2))<s_res) & P(1:i-1,3)==P(i,3),1);
                Ir = find((abs(P(1:i-1,1)-P(i,2))<s_res) & (abs(P(1:i-1,2)-P(i,1))<s_res) & P(1:i-1,3)==P(i,3),1);
                if ~isempty(I)
                    P(I,4:5) = P(I,4:5)+P(i,4:5);
                    P(i,:) = [];
                elseif ~isempty(Ir)
                    P(Ir,4:5) = P(Ir,4:5)+P(i,[5,4]);
                    P(i,:) = [];
                else
                    i = i+1;
                end
            end
            % Merge constant transitions:
            C = abs(P(:,4)-P(:,5))<p_res;
            if any(C)
                p = sum(P(C,4));
                P(C,:) = [];
                I = abs(P(:,4))>=p_res;
                p = p+sum(P(I,4));
                P(I,4:5) = P(I,4:5)-P(I,4);% For properties with constant first transition, make sure the fourth column is zero for all non-constant transitions
                %if abs(p)>=res(2) || isempty(P)
                P = [0,0,0,p,p;P];
                %end
            end
        end
        
        function P = select(P,e,ref,reg,strict)
            % Selects the transitions from P for which the endpoint (either
            % 1 for the transition's start or 2 for the transition's end)
            % is before (reg<0) or after (reg>0) the reference value ref.
            regions = [P(:,e)<ref,P(:,e)==ref,P(:,e)>ref];% Before, equal and after for transitions in positive direction
            I = regions(:,2+sign(reg));
            if ~strict
                I = I | regions(:,2);
            end
            P = P(I,:);
        end
        
        function P = selectDuplicates(P1,P2)
            P = [];
            for i=1:size(P1,1)
                if any(all(P1(i,:)==P2,2))
                    P(end+1,:) = P1(i,:);
                end
            end
        end
        
        function P = removeDuplicates(P)
            i=2;
            while i<=size(P,1)
                if any(all(P(1:i-1,:)==P(i,:),2))
                    P(i,:)=[];
                else
                    i = i+1;
                end
            end
        end
        
        function P = stepChange(P,f,t,pt)
            % Method used by stepwise properties to jump to the target
            % value pt at f and staying at this value till t, without
            % changing the values of the other step transitions before f
            % and after t.
            Pb = Property.select(P,2,f,-1,false);% All step transitions before (and including at) f
            Pa = Property.select(P,2,t,1,false);% All step transitions after (and including at) t
            po = Property.evaluate(P,f);% Original property value at f
            dp = pt-po;
            I = P(:,2)==f;
            if any(I)
                dp = dp-sum(P(I,5)-P(I,4));% If there is already a step transition at f, take it into account
            end
            I = P(:,2)==t;
            if ~any(I)
                Pa = Property.combine([t,t,0,0,0],Pa,1,1);% If there is not yet a step transition at t, insert a dummy one
            end
            Pb = Property.combine(Pb,[f,f,0,0,dp],1,1);% Include step transition to reach pt
            Pa = Property.shift(Pa,-dp);% Shift all transitions after (and including at) t by -dp
            P = Property.combine(Pb,Pa,1,1);% Recombine Pb and Pa
        end
    end
end

