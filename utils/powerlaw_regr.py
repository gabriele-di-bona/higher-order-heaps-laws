import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import scipy.odr
import scipy.stats

def powerLawRegr(seq):
    '''
        Calculates a power law regression of the function y(t) = seq[t], for t from 0.01*len(seq) to len(seq), as a linregress on the loglog scale.
        This function supposes that the sequence contains the values of y(t) for all integer t >= 0. First value usually is seq[0] = 0, so it is discarded in the funciton
        
        Returns slope, intercept, std_err of the loglog linregress
    '''
    xs_seq = np.arange(len(seq))+1
    ys_positive_seq = np.array([_ for _ in seq if _ > 0])
    xs_positive_seq = np.array([xs_seq[i] for i,_ in enumerate(seq) if _ > 0])
    len_positive_seq = len(ys_positive_seq)
    if len_positive_seq < 2:
        return 0,0,0
    else:
        # only retain the values from 0.01*len(seq) to len(seq)
        max_x = xs_positive_seq[-1]
        initial_x_regr = max(1,int(0.01*(max_x))) 
        xs_regr = np.array([_ for _ in xs_positive_seq if _ >= initial_t_regr])
        ys_regr = np.array([ys_positive_seq[i] for i,_ in enumerate(xs_positive_seq) if _ >= initial_x_regr])
        slope, intercept, r_value, p_value, std_err = linregress(np.log10(xs_regr), np.log10(ys_regr))
        return slope, intercept, std_err


def powerLawRegrPoints_old(xs, ys, add_intercept = False):
    '''
        Calculates a power law regression of the points xs and ys, which must be of the same size
        This is calculated as a linregress on the loglog scale of the points.
        
        Returns slope, intercept, std_err of the loglog linregress.
        If add_intercept is True, it also considers the intercept, otherwise it is forced to zero.
    '''
    if len(xs) < 2:
        return 0,0,0
    elif add_intercept == True:
        xs = np.log10(xs)
        ys = np.log10(ys)
        slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
        return slope, intercept, std_err
    else:
        # Our model is y = a * x, so things are quite simple, in this case...
        # x needs to be a column vector instead of a 1D vector for this, however.
        xs = np.log10(xs)
        ys = np.log10(ys)
        xs = xs[:,np.newaxis]
        slope, _, residuals, _ = np.linalg.lstsq(xs, ys, rcond=None)
        return slope[0], 0, residuals
    
def powerLawRegrPoints(xs, ys, intercept_left_lim = 0, intercept_right_lim = np.inf):
    '''
        Calculates a power law regression of the points xs and ys, which must be of the same size.
        This is calculated as a linregress on the loglog scale of the points.
        
        Returns slope, intercept, std_err of the loglog linregress.
        
        If intercept_left_lim == intercept_right_lim, the intercept (in loglog scale) is fixed to that value. If they are fixed to zero, then no value is considered.
        In general, they are the left and right bounds for the intercept. To have no bound, fix the parameter to either +-np.inf accordingly or None.
        
        The loglog intercept is returned as 10**intercept, so that it can be plotted directly in normal scale.
    '''
    if len(xs) < 2:
        if intercept_left_lim < intercept_right_lim:
            return 0, 10**0, np.array([0,0])
        else:
            return 0,10**0,0
    elif intercept_left_lim > intercept_right_lim:
        return 0,10**0,0
    elif intercept_left_lim == intercept_right_lim:
        # Our model is y = a * x + b, where b is fixed, so things are quite simple.
        # To make things easier, use y' = y - b
        # x needs to be a column vector instead of a 1D vector for this, however.
        def f(x,a):
            return a*x + intercept_left_lim
        xs = np.log10(xs)
        ys = np.log10(ys)
        popt, pcov = curve_fit(f, xs, ys, p0=[0])
        return popt[0], 10**intercept_left_lim, np.array([np.sqrt(np.diag(pcov))[0], 0])
    elif (intercept_left_lim == None or intercept_left_lim == -np.inf) and (intercept_right_lim == None or intercept_right_lim == np.inf):
        xs = np.log10(xs)
        ys = np.log10(ys)
        result = linregress(x, y)
        slope, intercept, r_value, p_value, std_err = result
        intercept_std_err = result.intercept_stderr
        return slope, 10**intercept, np.array([std_err, intercept_std_err])
    else:
        # Our model is y = a * x + b
        xs = np.log10(xs)
        ys = np.log10(ys)
#         xs = xs[:,np.newaxis]
        def f(x,a,b):
            return a*x + b
        # choose p0
        if intercept_left_lim == None or intercept_left_lim == -np.inf:
            p0_intercept = intercept_right_lim-1
        elif intercept_right_lim == None or intercept_right_lim == np.inf:
            p0_intercept = intercept_left_lim+1
        else:
            p0_intercept = (intercept_left_lim + intercept_right_lim) / 2
        popt, pcov = curve_fit(f, xs, ys, p0=[0,p0_intercept], bounds=([-np.inf,intercept_left_lim], [np.inf,intercept_right_lim]))
        return popt[0], 10**popt[1], np.sqrt(np.diag(pcov))

    
    
def lin_regr_with_stats(xs, ys, intercept_left_lim = 0, intercept_right_lim = np.inf, get_more_statistcs = False, alpha_for_conf_interv = 0.99):
    '''
        Calculates a linear regression of the points xs and ys, which must be of the same size.
        
        If get_more_statistcs == False, 
            returns slope, intercept, std_err (array including std_err_slope and std_err_intercept).
        
        If get_more_statistcs == True, 
            returns slope, intercept, std_err (array including std_err_slope and std_err_intercept), confidence_intervals, t_values, p_values.
        
        If intercept_left_lim == intercept_right_lim, the intercept (in loglog scale) is fixed to that value and so stderr is imposed to zero.
        In general, they are the left and right bounds for the intercept. To have no bound, fix the parameter to either +-np.inf accordingly or None.
    '''
    if len(xs) < 2 or intercept_left_lim > intercept_right_lim:
        if get_more_statistcs == False:
            return 0, 0, np.array([0,0])
        else:
            return 0, 0, np.array([0,0]), np.array([[0,0], [0,0]]), np.array([0,0]), np.array([0,0])
    elif intercept_left_lim == intercept_right_lim:
        # Our model is y = a * x + b, where b is fixed, so things are quite simple.
        # To make things easier, use y' = y - b
        # x needs to be a column vector instead of a 1D vector for this, however.
        def f(x,a):
            return a*x + intercept_left_lim
        parameters, pcov = curve_fit(f, x, y, p0=[0])
        if get_more_statistcs == False:
            return parameters[0], intercept_left_lim, np.array([np.sqrt(np.diag(pcov))[0], 0])
        else:
            def f_wrapper_for_odr(beta, x): # parameter order for odr (beta must be a vector of the parameters)
                return f(x, *beta)
            model = scipy.odr.odrpack.Model(f_wrapper_for_odr) # start the model with odr
            data = scipy.odr.odrpack.Data(x,y) # get the instance of the data
            myodr = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0) # apply the ODR using no iteration and starting from the parameters from curve_fit
            myodr.set_job(fit_type=2) # 2 is the ordinary least squares, the one used in curve_fit
            parameterStatistics = myodr.run() # get all the statistics
            df_e = len(x) - len(parameters) # degrees of freedom, error
            ci = [] # confidence intervals
            sd_beta = parameterStatistics.sd_beta # standard deviation of the error
            t_df = scipy.stats.t.ppf(alpha_for_conf_interv, df_e) # t-values at the alpha percentile with df_e degrees of freedom
            for i in range(len(parameters)):
                ci.append([parameters[i] - t_df * parameterStatistics.sd_beta[i], parameters[i] + t_df * parameterStatistics.sd_beta[i]])
            ci.append([0,0]) # for the absent intercept parameter
            tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
            pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values
            return parameters[0], intercept_left_lim, np.array([np.sqrt(np.diag(pcov))[0], 0]), np.array(ci), np.array(list(tstat_beta)+[0]), np.array(list(pstat_beta)+[0])
    else:
        # Our model is y = a * x + b
        def f(x,a,b):
            return a*x + b
        # choose p0
        if intercept_left_lim == None or intercept_left_lim == -np.inf:
            p0_intercept = intercept_right_lim-1
        elif intercept_right_lim == None or intercept_right_lim == np.inf:
            p0_intercept = intercept_left_lim+1
        else:
            p0_intercept = (intercept_left_lim + intercept_right_lim) / 2
        parameters, pcov = curve_fit(f, xs, ys, p0=[0,p0_intercept], bounds=([-np.inf,intercept_left_lim], [np.inf,intercept_right_lim]))
        if get_more_statistcs == False:
            return parameters[0], parameters[1], np.sqrt(np.diag(pcov))
        else:
            def f_wrapper_for_odr(beta, x): # parameter order for odr (beta must be a vector of the parameters)
                return f(x, *beta)
            model = scipy.odr.odrpack.Model(f_wrapper_for_odr) # start the model with odr
            data = scipy.odr.odrpack.Data(xs,ys) # get the instance of the data
            myodr = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0) # apply the ODR using no iteration and starting from the parameters from curve_fit
            myodr.set_job(fit_type=2) # 2 is the ordinary least squares, the one used in curve_fit
            parameterStatistics = myodr.run() # get all the statistics
            df_e = len(xs) - len(parameters) # degrees of freedom, error
            ci = [] # confidence intervals
            sd_beta = parameterStatistics.sd_beta # standard deviation of the error
            t_df = scipy.stats.t.ppf(alpha_for_conf_interv, df_e) # t-values at the alpha percentile with df_e degrees of freedom
            for i in range(len(parameters)):
                ci.append([parameters[i] - t_df * parameterStatistics.sd_beta[i], parameters[i] + t_df * parameterStatistics.sd_beta[i]])
            tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
            pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values
            return parameters[0], parameters[1], np.array([np.sqrt(np.diag(pcov))[0], 0]), np.array(ci), np.array(tstat_beta), np.array(pstat_beta)