function [ out ] = imas_gpr1d_fit_data(x_coord, y_coord, y_coord_err, varargin)
% ============================
% imas_gpr1d_fit_data function
% ============================
% Usage:
% ------
% out = imas_gpr1d_fit_data(x_coord, y_coord, y_coord_err)
% out = imas_gpr1d_fit_data(x_coord, y_coord, y_coord_err, x_coord_err)
%
% Fit Y profile as a function of X quantity
%
% Parameters
% ----------
% X_coordinate : 2D array size (points, time)
%     X coordinate of profile
%
% Y_coordinate : 2D array size (points, time)
%     Y coordinate of profile
%
% X_coordinate_errors : 2D array size (points, time)
%     X profile errors
%
% Y_coordinate_errors : 2D array size (points, time)
%     Y profile errors
%
% kernel_method : string (default='RQ_Kernel')
%     which kernel use for fit. One of
%
%       ``RQ_Kernel``
%         quadratic kernel for general purposes
%
%       ``Gibbs_Kernel``
%         IMPORTANT: use this kernel if profile contains a pedestal
%         with option optimise_all_params=True
%
% optimise_all_params : boolean (default=False)
%     If True optimise hyperparameters for all time slices in input (slow computation)
%
% slices_optim_nbr: int (default=10)
%     If optimise_all_params=False maximum number of slices equally spaced where to
%     perform search for optimised hyperparameters
%
% nbr_pts: int (default=100)
%     Number of points of fitting curve
%
% slices_nbr: int
%     Number of equally spaced slices in X and Y coordinate input arrays where fit is computed
%
% plot_fit: boolean (default=True)
%     If True saves in sub folders fitted curves images, and used when we have x_errors is not None
%
% Returns
% -------
% result : struct
%     with fields:
%
%         fit_x : size (points, time)
%             Fit x coordinate
%
%         fit_y : size (points, time)
%             Fit y coordinate
%
%         fit_y_error : size (points, time)
%             Fit y error
%
%         fit_dydx : size (points, time)
%             Fit derivative dy/dx
%
%         fit_dydx_y_error : size (points, time)
%             Fit y error of derivative dy/dx
%
%         x : size (points, time)
%             Original x coordinate data
%
%         y : size (points, time)
%             Original y coordinate data
%
%         x_error : size (points, time)
%             Original x errors data
%
%         y_error : size (points, time)
%             Original y errors data
%

% Inputs
% ------
% Input variables check:
% only want 3 optional inputs at most

% Parameters
% ----------

numvarargs = length(varargin);
if (numvarargs > 10)
    error('ERROR:imas_gpr1d_fit_data:TooManyInputs', ...
        'requires at most 13 inputs, for help type >>help imas_gpr1d_fit_data');
end
% If shot number empty or 0
if (nargin < 3)
    ME = MException('imas_gpr1d_fit_data:NoInputs', ...
     'ERROR: script requires at least x_coord, y_coord and y_coord_err as inputs');
    throw(ME);
end

% set defaults for optional inputs
optargs = {false 'RQ_Kernel' false 10 100 false false 0 0 0};

% now put these defaults into the valuesToUse cell array,
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[x_coord_err, kernel_method, optimise_all_params, slices_optim_nbr, ...
 nbr_pts, slices_nbr, plot_fit, dx_data, dy_data, dy_err] = optargs{:};

% Print
disp(' ')
fprintf('kernel_method = %s \n', kernel_method);
fprintf('Computing... \n\n');

[filepath] = fileparts(mfilename('fullpath'));

% Get current datetime
%tabDatetime = round(datevec(time));
%strDatetime = [num2str(tabDatetime(1)) num2str(tabDatetime(2:6), '%02d')];

% Generate random string to avoid file collisions
symbols = ['a':'z' 'A':'Z' '0':'9'];
MAX_ST_LENGTH = 20;
nums   = randi(numel(symbols), [1,  MAX_ST_LENGTH]);
stRand = symbols(nums);

fileRandomName    = ['tmpImas_gpr1d_fit_data', stRand, '.mat'];
outfileRandomName = ['out', fileRandomName];

save(fileRandomName, 'x_coord', 'y_coord', 'y_coord_err', ...
                     'x_coord_err', 'kernel_method', 'optimise_all_params', ...
                     'slices_optim_nbr', 'nbr_pts', 'slices_nbr', 'plot_fit', ...
                     'dx_data', 'dy_data', 'dy_err', '-v7');

cmdString = [' ', filepath, '/disk_fit_data.py ', fileRandomName];

try
    python_exe = '/Applications/Anaconda/python36/bin/python3.6';
    [status, outtmp] = unix([python_exe, cmdString]);
catch
    warning(['Problem with ', python_exe, ' trying single python command']);
    python_exe = 'python';
    [status, outtmp] = unix([python_exe, cmdString]);
end

% Check for errors in shell command
if nargout < 2 && status~=0
    delete(fileRandomName, outfileRandomName);
    error('ERROR:imas_gpr1d_fit_data:ExecutionError', outtmp);
end

load(outfileRandomName);

delete(fileRandomName, outfileRandomName);

% Out structure
out.fit_dydx = fit_dydx;
out.fit_dydx_x_error = fit_dydx_x_error;
out.fit_dydy_y_error = fit_dydy_y_error;
out.fit_x = fit_x;
out.fit_x_error = fit_x_error;
out.fit_y = fit_y;
out.fit_y_error = fit_y_error;
out.x = x;
out.x_error = x_error;
out.y = y;
out.y_error = y_error;

disp('End imas_gpr1d_fit_data')
end
