classdef Simulation < handle
    %SIMULATION Class used to setup the simulation environment and perform
    %simulation steps.
    
    properties
        Sc      % Scenario in which all vehicles move
        Cars    % Array of all vehicles that are part of this simulation
        k = 0   % Current time step
        dt = 0.1% Sample time
        
        useMEX = false
    end
    
    properties(Access=private)
        UI
        mexHandle = []
        customDrivers
        colors
        
        setupUICallback = @(varargin)[];
        updateUICallback = @(varargin)[];
    end
    
    methods
        function S = Simulation(config,scenarioName,vehicleTypes,useMEX)
            %SIMULATION Construct an instance of this class
            if nargin>3
                S.useMEX = useMEX;
            end
            
            S.dt = config.dt;% Note: other configuration options are not supported in pure Matlab version
            S.Sc = Scenario(scenarioName);
            S.Cars = Vehicle.empty();
            nbCars = sum([vehicleTypes.amount]);
            S.colors = zeros(nbCars,3);
            if S.useMEX
                % Loop over vehicleTypes and take care of custom driving
                % policies:
                id = 0;
                for t=1:numel(vehicleTypes)
                    ids = id:id+vehicleTypes(t).amount-1;
                    if ~isstring(vehicleTypes(t).policy)
                        S.customDrivers(end+1) = struct("policy",vehicleTypes(t).policy,"ids",ids);
                        S.colors(ids+1,:) = vehicleTypes(t).policy.color;
                        vehicleTypes(t).policy = "custom";
                    else
                        switch(vehicleTypes(t).policy)
                            case "slow"
                                S.colors(ids+1,:) = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.SLOW)).color;
                            case "normal"
                                S.colors(ids+1,:) = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.DEFENSIVE)).color;
                            case "fast"
                                S.colors(ids+1,:) = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.AGGRESSIVE)).color;
                        end
                    end
%                     for i=ids
%                         S.Cars(end+1) = Vehicle(struct(),[],models.KinematicBicycleModel(
                    id = id+vehicleTypes(t).amount;
                end
                S.mexHandle = mexSim('new',config,scenarioName,vehicleTypes);
            else
                % Initialize new cars randomly based on the given
                % parameters
                
                % Calculate start positions of all vehicles. Vehicles will be
                % placed equidistantly and occupy all available lanes.
                [MR,ML,Ms] = S.Sc.getRoadMapping();
                length = MR(end,1);
                cPerm = randperm(nbCars);
                c = 1;
                for t=1:numel(vehicleTypes)
                    for i=1:vehicleTypes(t).amount
                        % Use road and lane mappings to determine the
                        % locations of the vehicles along the road:
                        p = (cPerm(c)-0.5+rand()/2-0.25)*length/nbCars;% Equally spaced position variable (randomly perturbated), used to evaluate MR, ML and Ms
                        R = Property.evaluate(MR,p);
                        L = Property.evaluate(ML,p);
                        s = Property.evaluate(Ms,p);
                        l = S.Sc.roads(R).evalProp('offset',s,L);
                        %TODO: create separate lateral controller for trucks (with larger
                        %lengths)
                        min_size = vehicleTypes(t).sizeBounds(:,1);
                        max_size = vehicleTypes(t).sizeBounds(:,2);
                        cgr = 0.45;% CG ratio 0.6
                        v_size = rand(3,1).*(max_size-min_size)+min_size;
                        v0 = (0.7+rand()*0.3)*S.Sc.roads(R).evalProp('speed',s,L);
                        vehicle_props = struct('road',R,'pos',[s;l],'vel',[v0;0;0],'heading',0);
                        switch(vehicleTypes(t).policy)
                            case "step"
                                driver = drivers.StepDriver();
                            case "slow"
                                driver = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.SLOW));
                            case "normal"
                                driver = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.DEFENSIVE));
                            case "fast"
                                driver = drivers.ManualDriver(struct('driver_type',drivers.ManualDriver.AGGRESSIVE));
                            otherwise
                                driver = vehicleTypes(t).policy;
                        end
                        S.colors(c,:) = driver.color;
                        model = models.KinematicBicycleModel('Size',v_size,'RelCG',[cgr;0]);
                        S.Cars(c) = Vehicle(vehicle_props,model,driver,S.Sc);
                        c = c+1;
                    end
                end
                S.updateStates();
            end
        end
        
        function delete(S)
            % Destroy C++ object:
            if ~isempty(S.mexHandle)
                mexRoad('delete',S.mexHandle);
            end
            S.mexHandle = [];
            
            % Close visualisation if any:
            if ~isempty(S.UI) && ishghandle(S.UI.f)
                close(S.UI.f);
            end
            S.UI = [];
        end
        
        function stop = step(S)
            %Perform 1 simulation step, returns true if simulation was
            %stopped due to a collision
            tic;
            if ~isempty(S.mexHandle)
                % Loop over custom drivers and evaluate their policies
                % before taking the step
                for c=1:numel(S.customDrivers)
                    for id=S.customDrivers(c).ids
                        state = mexSim('getVehicle',S.mexHandle,'policy',id);
                        mexSim('setActions',S.mexHandle,id,S.customDrivers(c).policy.drive(state));
                    end
                end
                stop = mexSim('step',S.mexHandle);
            else
                % Advance time for each vehicle:
                for v=1:numel(S.Cars)
                    S.Cars(v).advanceTime(S.dt);
                end
                % Update augmented states of all vehicles and check for
                % collisions:
                stop = S.updateStates();
                S.k = S.k+1;
            end
            disp(compose("Iteration %d (took %2.3f seconds) ; Simulation time: %4.2f",S.k,toc,S.k*S.dt));
        end
        
        function [state,input] = getVehicleInfo(S,V,info)
            if ~isempty(S.mexHandle)
                [state,input] = mexSim(S.mexHandle,"getVehicle",V,info);
            else
                switch(info)
                    case "model"
                        state = [S.Cars(V).pos;S.Cars(V).ang;S.Cars(V).vel;S.Cars(V).ang_vel];
                        input = S.Cars(V).input;
                    case "policy"
                        state = S.Cars(V).state;
                        input = S.Cars(V).actions;
                    case "road"
                        state = S.Cars(V).roadInfo;
                        input = S.Cars(V).colStatus;
                end
            end
        end
        
        function customizeUI(S,setupCallback,updateCallback)
            S.setupUICallback = setupCallback;
            S.updateUICallback = updateCallback;
        end
        
        function visualize(S)
            redrawColors = false;
            if isempty(S.UI) || ~ishghandle(S.UI.f)
                S.setupVisualisation();
                redrawColors = true;
            end
            
            for v=1:numel(S.Cars)
                if redrawColors
                    ec = [0,0,0];
                    fc = S.colors(v,:);
                    if v==S.UI.V
                        ec = [0.929,0.789,0.076];
                    end
                else
                    ec = [];
                    fc = [];
                end
                S.Cars(v).plot(S.UI.dAxes,fc,ec);
            end
            S.UI.dPatch.Vertices = utils.getCuboidPatch(S.Cars(S.UI.V).pos(1:2),ones(2,2)*S.UI.detail.W/2,S.Cars(S.UI.V).ang(1));
            camup(S.UI.dAxes,[cos(S.Cars(S.UI.V).ang(1)),sin(S.Cars(S.UI.V).ang(1)),1]);
            xlim(S.UI.dAxes,S.Cars(S.UI.V).pos(1)+[-1,1]*S.UI.detail.W/2);
            ylim(S.UI.dAxes,S.Cars(S.UI.V).pos(2)+[-1,1]*S.UI.detail.W/2);
            zlim(S.UI.dAxes,S.Cars(S.UI.V).pos(3)+[-1,1]*S.UI.detail.H);

            for i=1:numel(S.UI.info.names)
                S.UI.info.lines.(S.UI.info.names{i}).XData(S.k+1) = S.k*S.dt;% Set x data
            end
            % Set y data
            S.UI.info.lines.VelRef.YData(S.k+1) = S.Cars(S.UI.V).actions(1);
            S.UI.info.lines.OffRef.YData(S.k+1) = S.Cars(S.UI.V).state(1)+S.Cars(S.UI.V).actions(2);
            S.UI.info.lines.Vel.YData(S.k+1) = S.Cars(S.UI.V).vel(1);
            S.UI.info.lines.Off.YData(S.k+1) = S.Cars(S.UI.V).state(1);
%             S.UI.info.lines.FrontVel.YData(S.k+1) = sim_data(S.UI.V).driverState(1,I);
%             S.UI.info.lines.FrontOff.YData(S.k+1) = sim_data(S.UI.V).driverState(2,I);
%             S.UI.info.lines.RightFree.YData(S.k+1) = sim_data(S.UI.V).driverState(3,I);
%             S.UI.info.lines.LeftFree.YData(S.k+1) = sim_data(S.UI.V).driverState(4,I);

            % Call custom UI update
            S.updateUICallback(S.UI,S);
            
            xlim(S.UI.vel_axes,[max(S.k*S.dt-100,0),S.k*S.dt+5]);
            xlim(S.UI.off_axes,[max(S.k*S.dt-100,0),S.k*S.dt+5]);
            xlim(S.UI.extra1_axes,[max(S.k*S.dt-100,0),S.k*S.dt+5]);
            xlim(S.UI.extra2_axes,[max(S.k*S.dt-100,0),S.k*S.dt+5]);
            %S.UI.stepSlider.Value = S.k;
            S.UI.timeLabel.String = compose("k = %d ; t = %4.2fs",S.k,S.k*S.dt);
            drawnow('limitrate');
        end
    end
    
    methods (Access=private)
        function col = updateStates(S)
            col = false;
            % First calculate relative distances between all vehicles:
            positions = [S.Cars.pos];
            D = squareform(pdist(positions(1:2,:)'));
            roadInfo = [S.Cars.roadInfo];

            % Mask denoting which other vehicles are in the neighbourhood
            % of EV
            M = D < Vehicle.D_max & (1-eye(numel(S.Cars)));% OV should be within the detection horizon
            M = M & [roadInfo.direction]==[roadInfo.direction]';% And have the same direction as EV
            for v=1:numel(S.Cars)
                % Only select N_max closest vehicles
                N = find(M(:,v));
                [~,ids] = mink(D(M(:,v),v),Vehicle.N_max);
                N = N(ids);% Neighbour indices
                offsets = S.Sc.getRoadOffsets(S.Cars(v),S.Cars(N));
                % Update the augmented state for this vehicle:
                S.Cars(v).updateDriverState(roadInfo(N),D(N,v),offsets);
                % Collision handling:
                if S.Cars(v).col_status~=Vehicle.COL_NONE
                    switch(S.Cars(v).col_status)
                        case Vehicle.COL_LEFT
                            reason = 'the left road boundary';
                        case Vehicle.COL_RIGHT
                            reason = 'the right road boundary';
                        otherwise
                            reason = ['vehicle ',num2str(N(S.Cars(v).col_status))];
                    end
                    disp(['Vehicle ',num2str(v),' collided with ',reason,'.']);
                    col = true;
                end
            end
        end
        
        function setupVisualisation(S)
            %%% Set up figure:
            S.UI = struct();
            S.UI.f = figure(1);
            S.UI.f.WindowState = "maximized";
            S.UI.detail = struct("W",50,"H",5);
            S.UI.V = 1;

            % add a scenario plot 
            S.UI.scAxes = subplot(4,2,[1,3]);
            S.Sc.plot(S.UI.scAxes,"Lanes","off","Flatten","on","OutlineColor",[0.6,0.6,0.6]);% Plot the road
            [dpV,dpF] = utils.getCuboidPatch([0;0],ones(2,2)*S.UI.detail.W/2,0);
            % Plot the detail patch
            S.UI.dPatch = patch(S.UI.scAxes,"Vertices",dpV,"Faces",dpF,"FaceColor","none","EdgeColor",[0.635,0.078,0.184],"LineWidth",2);
            daspect(S.UI.scAxes,[1,1,1]);
            title(S.UI.scAxes,'Scenario plot');
            xlabel(S.UI.scAxes,'X');
            ylabel(S.UI.scAxes,'Y');

            % add a detail plot
            S.UI.dPanel = uipanel(S.UI.f,'Position',[0.51,0.51,0.49,0.39],'Units','Normal');
            S.UI.dPanel.Title = 'Detail plot';
            S.UI.dPanel.TitlePosition = 'centertop'; 
            S.UI.dPanel.FontSize = 11;
            S.UI.dPanel.FontWeight = 'bold';
            S.UI.dAxes = axes(S.UI.dPanel);
            S.Sc.plot(S.UI.dAxes);% Plot the road
            daspect(S.UI.dAxes,[1,1,1]);
            axis(S.UI.dAxes,'off');
            S.UI.dAxes.CameraViewAngleMode = 'manual';
            S.UI.dAxes.Clipping = 'off';

            % add a control panel
            S.UI.cPanel = uipanel(S.UI.f,'Position',[0.51,0.90,0.49,0.1],'Units','Normal');
            S.UI.cPanel.Title = 'Replay control';
            S.UI.cPanel.TitlePosition = 'centertop';
            S.UI.cPanel.FontSize = 11;
            S.UI.cPanel.FontWeight = 'bold';
            uicontrol(S.UI.cPanel,'Style','text','Units','normalized','Position',[0.29,0.75,0.2,0.2],'String','Displayed vehicle:','HorizontalAlignment','right');
            S.UI.vehicleEdit = uicontrol(S.UI.cPanel,'Style','edit','Units','normalized','Position',[0.5,0.75,0.1,0.2],'String',num2str(S.UI.V),'Callback',@handleReplayControl);
            S.UI.playBtn = uicontrol(S.UI.cPanel,'Style','pushbutton','Units','normalized','Position',[0.45,0.45,0.1,0.2],'String','Pause','Callback',@handleReplayControl);
            S.UI.timeLabel = uicontrol(S.UI.cPanel,'Style','text','Units','normalized','Position',[0.4,0.22,0.2,0.2],'String',compose("k = %d ; t = %4.2fs",0,0));
            %S.UI.stepSlider = uicontrol(S.UI.cPanel,'Style','slider','Units','normalized','Position',[0.02,0.02,0.96,0.2],'SliderStep',[1/(config.k_end-config.k_0),10/(config.k_end-config.k_0)],'Min',config.k_0,'Max',config.k_end,'Value',config.k_0,'Callback',@handleReplayControl);

            % add vehicle info plot
            subplot(S.UI.scAxes);
            S.UI.vel_axes = subplot(4,2,5);
            S.UI.info.lines = struct();
            %nans = nan(1,1+config.k_end-config.k_0);
            nans = nan(1,1000);
            S.UI.info.lines.VelRef = line(S.UI.vel_axes,nans,nans,'Color','r','LineStyle','--');
            S.UI.info.lines.Vel = line(S.UI.vel_axes,nans,nans,'Color','b');
            title(S.UI.vel_axes,'Vehicle info');
            ylabel(S.UI.vel_axes,'m/s');

            S.UI.off_axes = subplot(4,2,7);
            S.UI.info.lines.OffRef = line(S.UI.off_axes,nans,nans,'Color','r','LineStyle','--');
            S.UI.info.lines.Off = line(S.UI.off_axes,nans,nans,'Color','b');
            ylabel(S.UI.off_axes,'m');
            xlabel(S.UI.off_axes,'t (s)');

            % add two extra plots that can be customized:
            S.UI.extra1_axes = subplot(4,2,6);
            S.UI.extra2_axes = subplot(4,2,8);
            
            % Call custom UI setup
            S.UI = S.setupUICallback(S.UI);

            S.UI.info.names = fieldnames(S.UI.info.lines);
        end
    end
end

