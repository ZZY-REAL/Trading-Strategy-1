from ssquant.backtest.backtest_core import MultiSourceBacktester
import pandas as pd
import numpy as np
from ssquant.api.strategy_api import StrategyAPI

"""
高级跨品种套利策略 - 深度优化版本

主要改进：
1. Z-Score 过滤器：防止在价差处于历史极端位置时进行趋势跟踪开仓。
2. ATR 动态止损：使用价差比率的平均真实波幅（ATR）设定止损。
3. 波动率平衡（Beta Neutral）：根据两个品种的相对波动率动态调整对手手数。
4. 异常波动过滤：识别并过滤价差突变（Spike Filter）。
5. 变量名优化：提高代码可读性与维护性。
"""

# 全局策略状态
strategy_state = {
    'long_entry_ratio': 0,
    'active_trailing_stop': 0,
    'short_entry_ratio': 0,
    'bars_since_entry_0': 0,
    'bars_since_entry_1': 0,
    'last_entry_bar_idx': -1,
    'last_pos_0': 0,
    'last_pos_1': 0,
    
    # 指标历史记录
    'ratio_history': [],
    'smooth_range_history': [],
    'filter_line_history': [],
    'trend_up_count_history': [],
    'trend_down_count_history': [],
    'atr_history': [],
    'z_score_history': [],
    
    'last_calculated_bar_idx': -1,
    
    # 指标计算状态 (EMA/Volatility)
    'indicator_ema_state': {
        'avg_range_fast': None,
        'smooth_avg_range_fast': None,
        'avg_range_slow': None,
        'smooth_avg_range_slow': None,
        'atr_val': None,
        'volatility_0': None,
        'volatility_1': None
    }
}

def initialize(api: StrategyAPI):
    """策略初始化"""
    api.log("高级跨品种套利策略启动...")
    api.log("包含 Z-Score、ATR 止损与波动率平衡优化")

def calculate_indicators_incremental(current_ratio, ratio_high, ratio_low, previous_ratio, params, ema_state):
    """增量计算所有技术指标"""
    global strategy_state
    
    # 1. 基础 RangeFilter 逻辑
    ratio_change = abs(current_ratio - previous_ratio)
    
    # 第一层 EMA (快速周期)
    alpha_fast = 2.0 / (params['fast_period'] + 1)
    if ema_state['avg_range_fast'] is None: 
        ema_state['avg_range_fast'] = ratio_change
    else: 
        ema_state['avg_range_fast'] = alpha_fast * ratio_change + (1 - alpha_fast) * ema_state['avg_range_fast']
    
    alpha_fast_smooth = 2.0 / (params['fast_period'] * 2)
    if ema_state['smooth_avg_range_fast'] is None: 
        ema_state['smooth_avg_range_fast'] = ema_state['avg_range_fast']
    else: 
        ema_state['smooth_avg_range_fast'] = alpha_fast_smooth * ema_state['avg_range_fast'] + (1 - alpha_fast_smooth) * ema_state['smooth_avg_range_fast']
    
    smooth_range_fast = ema_state['smooth_avg_range_fast'] * params['fast_range_mult']
    
    # 第二层 EMA (慢速周期)
    alpha_slow = 2.0 / (params['fast_period'] * 3 + 1)
    if ema_state['avg_range_slow'] is None: 
        ema_state['avg_range_slow'] = ratio_change
    else: 
        ema_state['avg_range_slow'] = alpha_slow * ratio_change + (1 - alpha_slow) * ema_state['avg_range_slow']
    
    alpha_slow_smooth = 2.0 / (params['fast_period'] * 3 * 2)
    if ema_state['smooth_avg_range_slow'] is None: 
        ema_state['smooth_avg_range_slow'] = ema_state['avg_range_slow']
    else: 
        ema_state['smooth_avg_range_slow'] = alpha_slow_smooth * ema_state['avg_range_slow'] + (1 - alpha_slow_smooth) * ema_state['smooth_avg_range_slow']
    
    smooth_range_slow = ema_state['smooth_avg_range_slow'] * params['fast_range_mult'] * 0.6
    
    combined_smooth_range = (smooth_range_fast + smooth_range_slow) / 2
    
    # 范围过滤器线 (Range Filter Line)
    prev_filter_line = strategy_state['filter_line_history'][-1] if strategy_state['filter_line_history'] else current_ratio
    if current_ratio > prev_filter_line:
        current_filter_line = prev_filter_line if current_ratio - combined_smooth_range < prev_filter_line else current_ratio - combined_smooth_range
    else:
        current_filter_line = prev_filter_line if current_ratio + combined_smooth_range > prev_filter_line else current_ratio + combined_smooth_range
        
    # 趋势强度计数
    prev_trend_up = strategy_state['trend_up_count_history'][-1] if strategy_state['trend_up_count_history'] else 0
    prev_trend_down = strategy_state['trend_down_count_history'][-1] if strategy_state['trend_down_count_history'] else 0
    
    if current_filter_line > prev_filter_line:
        trend_up, trend_down = prev_trend_up + 1, 0
    elif current_filter_line < prev_filter_line:
        trend_up, trend_down = 0, prev_trend_down + 1
    else:
        trend_up, trend_down = prev_trend_up, prev_trend_down

    # 2. ATR 计算 (用于动态止损)
    true_range = max(ratio_high - ratio_low, abs(ratio_high - previous_ratio), abs(ratio_low - previous_ratio))
    alpha_atr = 2.0 / (params['atr_period'] + 1)
    if ema_state['atr_val'] is None: 
        ema_state['atr_val'] = true_range
    else: 
        ema_state['atr_val'] = alpha_atr * true_range + (1 - alpha_atr) * ema_state['atr_val']
    
    # 3. Z-Score 计算 (用于极端位置过滤)
    historical_ratios = np.array(strategy_state['ratio_history'][-params['z_score_period']:])
    if len(historical_ratios) >= params['z_score_period']:
        mean_ratio = np.mean(historical_ratios)
        std_ratio = np.std(historical_ratios)
        z_score_val = (current_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
    else:
        z_score_val = 0
        
    return combined_smooth_range, current_filter_line, trend_up, trend_down, ema_state['atr_val'], z_score_val

def calculate_volatility_balanced_volumes(api, params, asset0_close, asset1_close):
    """基于两个品种的相对波动率计算平衡的手数比例"""
    kline0 = api.get_klines(0)['close']
    kline1 = api.get_klines(1)['close']
    
    # 计算最近20周期的收益率标准差作为波动率
    volatility_0 = kline0.pct_change().iloc[-20:].std()
    volatility_1 = kline1.pct_change().iloc[-20:].std()
    
    if pd.isna(volatility_0) or pd.isna(volatility_1) or volatility_0 == 0 or volatility_1 == 0:
        return params['default_volume_0'], params['default_volume_1']
    
    # 波动率比率 (Beta Neutral 思路)
    vol_ratio = volatility_0 / volatility_1
    
    # 计算基准手数
    if params['use_auto_lots']:
        base_volume = max(1, int(params['total_capital'] / (asset0_close * 100 * 0.1))) # 假设10%保证金
    else:
        base_volume = params['default_volume_0']
        
    # 对冲品种的手数根据波动率反向调整
    adjusted_volume_1 = max(1, int(base_volume * vol_ratio))
    
    return base_volume, adjusted_volume_1

def arbitrage_strategy(api: StrategyAPI):
    """核心策略逻辑"""
    global strategy_state
    
    if not api.require_data_sources(2): return
    
    # 1. 策略参数加载
    if 'params_cached' not in strategy_state:
        strategy_state['params'] = {
            'use_auto_lots': api.get_param('lotskg', True),
            'default_volume_0': api.get_param('lots_data1', 1),
            'default_volume_1': api.get_param('lots_data2', 1),
            'total_capital': api.get_param('fund', 20000),
            'fast_period': api.get_param('fast_period', 50),
            'fast_range_mult': api.get_param('fast_range', 1.5),
            'trend_threshold': api.get_param('x_filter', 3),
            'atr_stop_mult': api.get_param('stop_mult', 3.0),
            'atr_period': api.get_param('atr_period', 20),
            'z_score_period': api.get_param('z_period', 100),
            'z_score_threshold': api.get_param('z_threshold', 2.0),
            'max_spike_percent': api.get_param('max_spike', 0.03)
        }
        strategy_state['params_cached'] = True
    
    params = strategy_state['params']
    bar_idx = api.get_idx(0)
    if bar_idx < params['z_score_period']: return

    # 2. 实时数据获取
    kline0 = api.get_klines(0).iloc[bar_idx]
    kline1 = api.get_klines(1).iloc[bar_idx]
    
    close_0, close_1 = kline0['close'], kline1['close']
    current_ratio = close_0 / close_1
    
    # 计算价差比率的高低点 (考虑同步性)
    ratio_high = max(kline0['high']/kline1['low'], kline0['open']/kline1['open'], current_ratio)
    ratio_low = min(kline0['low']/kline1['high'], kline0['open']/kline1['open'], current_ratio)
    
    previous_ratio = strategy_state['ratio_history'][-1] if strategy_state['ratio_history'] else current_ratio
    
    # 3. 突变过滤 (异常值检测)
    if abs(current_ratio / previous_ratio - 1) > params['max_spike_percent']:
        api.log(f"跳过异常波动: {abs(current_ratio/previous_ratio-1):.2%}")
        return

    # 4. 指标增量更新
    smooth_range, filter_line, trend_up, trend_down, atr_val, z_score_val = calculate_indicators_incremental(
        current_ratio, ratio_high, ratio_low, previous_ratio, params, strategy_state['indicator_ema_state']
    )
    
    # 更新历史状态
    strategy_state['ratio_history'].append(current_ratio)
    strategy_state['filter_line_history'].append(filter_line)
    strategy_state['trend_up_count_history'].append(trend_up)
    strategy_state['trend_down_count_history'].append(trend_down)
    strategy_state['atr_history'].append(atr_val)
    
    # 5. 仓位与信号逻辑
    pos_0 = api.get_pos(0)
    pos_1 = api.get_pos(1)
    
    # 进场信号：趋势确认 + Z-Score 风险过滤
    long_signal = (current_ratio > filter_line and trend_up > params['trend_threshold'] and z_score_val < params['z_score_threshold'])
    short_signal = (current_ratio < filter_line and trend_down > params['trend_threshold'] and z_score_val > -params['z_score_threshold'])
    
    if pos_0 == 0 and pos_1 == 0:
        if long_signal:
            vol_0, vol_1 = calculate_volatility_balanced_volumes(api, params, close_0, close_1)
            api.buy(volume=vol_0, order_type='next_bar_open', index=0)
            api.sellshort(volume=vol_1, order_type='next_bar_open', index=1)
            strategy_state['active_trailing_stop'] = current_ratio - atr_val * params['atr_stop_mult']
            strategy_state['last_entry_bar_idx'] = bar_idx
            api.log(f"多头开仓 | Z-Score: {z_score_val:.2f} | 初始止损: {strategy_state['active_trailing_stop']:.4f}")
            
        elif short_signal:
            vol_0, vol_1 = calculate_volatility_balanced_volumes(api, params, close_0, close_1)
            api.sellshort(volume=vol_0, order_type='next_bar_open', index=0)
            api.buy(volume=vol_1, order_type='next_bar_open', index=1)
            strategy_state['active_trailing_stop'] = current_ratio + atr_val * params['atr_stop_mult']
            strategy_state['last_entry_bar_idx'] = bar_idx
            api.log(f"空头开仓 | Z-Score: {z_score_val:.2f} | 初始止损: {strategy_state['active_trailing_stop']:.4f}")

    else:
        # 6. ATR 追踪止损更新
        if pos_0 > 0: # 正在做多价差
            potential_new_stop = current_ratio - atr_val * params['atr_stop_mult']
            strategy_state['active_trailing_stop'] = max(strategy_state['active_trailing_stop'], potential_new_stop)
            
            if current_ratio < strategy_state['active_trailing_stop'] and (bar_idx > strategy_state['last_entry_bar_idx']):
                api.sell(order_type='next_bar_open', index=0)
                api.buycover(order_type='next_bar_open', index=1)
                api.log(f"多头ATR止损 | 价格: {current_ratio:.4f} | 止损位: {strategy_state['active_trailing_stop']:.4f}")
                
        elif pos_0 < 0: # 正在做空价差
            potential_new_stop = current_ratio + atr_val * params['atr_stop_mult']
            strategy_state['active_trailing_stop'] = min(strategy_state['active_trailing_stop'], potential_new_stop)
            
            if current_ratio > strategy_state['active_trailing_stop'] and (bar_idx > strategy_state['last_entry_bar_idx']):
                api.buycover(order_type='next_bar_open', index=0)
                api.sell(order_type='next_bar_open', index=1)
                api.log(f"空头ATR止损 | 价格: {current_ratio:.4f} | 止损位: {strategy_state['active_trailing_stop']:.4f}")

if __name__ == "__main__":
    pass
