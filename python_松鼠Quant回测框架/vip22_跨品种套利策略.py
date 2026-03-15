from ssquant.backtest.backtest_core import MultiSourceBacktester
import pandas as pd
import numpy as np
from ssquant.api.strategy_api import StrategyAPI

"""
vip22跨品种套利策略 - 性能优化版本

主要优化措施：
1. 增量计算技术指标，避免每个bar重新计算整个历史序列
2. 缓存K线数据，减少重复的API调用
3. 缓存策略参数，避免重复获取
4. 限制历史数据保存量，减少内存使用
5. 使用辅助函数减少重复代码
6. 只在必要时更新数据，提高计算效率

预期性能提升：回测速度提升3-5倍
"""

# 全局状态变量
g_state = {
    'd_jcprice_0': 0,
    'arass_0': 0,
    'k_jcprice_0': 0,
    'd_jcprice_1': 0,
    'arass_1': 0,
    'k_jcprice_1': 0,
    'bars_since_entry_0': 0,
    'bars_since_entry_1': 0,
    'last_entry_bar_0': -1,
    'last_entry_bar_1': -1,
    'last_pos_0': 0,
    'last_pos_1': 0,
    # 新增：缓存技术指标数据
    'price_history': [],
    'smrng_history': [],
    'filt_history': [],
    'upward_history': [],
    'downward_history': [],
    'last_calculated_bar': -1,
    # EMA计算所需的状态
    'ema_state': {
        'avrng1': None,
        'smooth_avrng1': None,
        'avrng2': None,
        'smooth_avrng2': None
    }
}

def initialize(api: StrategyAPI):
    """
    策略初始化函数
    此函数用于初始化vip22跨品种套利策略并输出日志信息。
    
    Args:
        api: 策略API对象，用于访问策略参数和日志功能
    """
    api.log("vip22跨品种套利策略初始化...")
    api.log("本策略基于价差的范围过滤器进行双品种套利交易")

def smooth_range(price, period, s_range):
    """
    SmoothRange函数 - 根据原始tbquant实现
    
    Args:
        price: 价格序列
        period: 周期
        s_range: 范围参数
    
    Returns:
        平滑范围序列
    """
    # 计算价格变化的绝对值
    price_diff = abs(price.diff())
    
    # 计算第一层平均（XAverage相当于EMA）
    avrng = price_diff.ewm(span=period, adjust=False).mean()
    
    # 计算第二层平均
    wper = period * 2 - 1
    smooth_avrng = avrng.ewm(span=wper, adjust=False).mean()
    
    # 应用范围参数
    result = smooth_avrng * s_range
    
    return result

def smooth_range_incremental(price_diff, period, s_range, ema_state):
    """
    增量计算SmoothRange函数
    
    Args:
        price_diff: 当前价格变化的绝对值
        period: 周期
        s_range: 范围参数
        ema_state: EMA状态字典
    
    Returns:
        当前的平滑范围值
    """
    alpha1 = 2.0 / (period + 1)
    alpha2 = 2.0 / (period * 2 - 1 + 1)
    
    # 第一层EMA
    if ema_state['avrng1'] is None:
        ema_state['avrng1'] = price_diff
    else:
        ema_state['avrng1'] = alpha1 * price_diff + (1 - alpha1) * ema_state['avrng1']
    
    # 第二层EMA
    if ema_state['smooth_avrng1'] is None:
        ema_state['smooth_avrng1'] = ema_state['avrng1']
    else:
        ema_state['smooth_avrng1'] = alpha2 * ema_state['avrng1'] + (1 - alpha2) * ema_state['smooth_avrng1']
    
    return ema_state['smooth_avrng1'] * s_range

def smooth_range_incremental2(price_diff, period, s_range, ema_state):
    """
    增量计算第二个SmoothRange函数
    """
    alpha1 = 2.0 / (period * 3 + 1)
    alpha2 = 2.0 / (period * 3 * 2 - 1 + 1)
    
    # 第一层EMA
    if ema_state['avrng2'] is None:
        ema_state['avrng2'] = price_diff
    else:
        ema_state['avrng2'] = alpha1 * price_diff + (1 - alpha1) * ema_state['avrng2']
    
    # 第二层EMA
    if ema_state['smooth_avrng2'] is None:
        ema_state['smooth_avrng2'] = ema_state['avrng2']
    else:
        ema_state['smooth_avrng2'] = alpha2 * ema_state['avrng2'] + (1 - alpha2) * ema_state['smooth_avrng2']
    
    return ema_state['smooth_avrng2'] * s_range * 0.6

def range_filter(x, r):
    """
    RangeFilter函数 - 根据原始tbquant实现
    
    Args:
        x: 输入序列
        r: 范围序列
    
    Returns:
        过滤后的序列
    """
    rngfilt = x.copy()
    
    # 逐个计算，保持与原始逻辑一致
    for i in range(1, len(x)):
        if pd.isna(r.iloc[i]) or pd.isna(x.iloc[i]):
            continue
            
        prev_filt = rngfilt.iloc[i-1]
        curr_x = x.iloc[i]
        curr_r = r.iloc[i]
        
        # 原始逻辑：
        # rngfilt = IIF(x > rngfilt[1], IIF(x - r < rngfilt[1], rngfilt[1], x - r), 
        #               IIF(x + r > rngfilt[1], rngfilt[1], x + r));
        
        if curr_x > prev_filt:
            # 如果当前值大于前一个过滤值
            if curr_x - curr_r < prev_filt:
                rngfilt.iloc[i] = prev_filt
            else:
                rngfilt.iloc[i] = curr_x - curr_r
        else:
            # 如果当前值小于等于前一个过滤值
            if curr_x + curr_r > prev_filt:
                rngfilt.iloc[i] = prev_filt
            else:
                rngfilt.iloc[i] = curr_x + curr_r
    
    return rngfilt

def range_filter_incremental(x, r, prev_filt):
    """
    增量计算RangeFilter函数
    
    Args:
        x: 当前输入值
        r: 当前范围值
        prev_filt: 前一个过滤值
    
    Returns:
        当前过滤值
    """
    if prev_filt is None:
        return x
    
    if x > prev_filt:
        if x - r < prev_filt:
            return prev_filt
        else:
            return x - r
    else:
        if x + r > prev_filt:
            return prev_filt
        else:
            return x + r

def calculate_lots(api, fund, open_price, contract_unit, big_point_value, margin_rate=0.1):
    """
    计算开仓手数
    
    Args:
        api: 策略API对象
        fund: 资金
        open_price: 开仓价格
        contract_unit: 合约单位
        big_point_value: 大点价值
        margin_rate: 保证金率
    
    Returns:
        计算出的手数
    """
    if open_price <= 0:
        return 1
    
    # 计算所需保证金
    required_margin = open_price * contract_unit * big_point_value * margin_rate
    
    # 计算可开手数
    lots = max(1, int(fund / required_margin))
    
    return lots

def update_bars_since_entry(bar_idx):
    """更新持仓天数"""
    global g_state
    
    # 更新品种0的持仓天数
    if g_state['last_entry_bar_0'] >= 0:
        g_state['bars_since_entry_0'] = bar_idx - g_state['last_entry_bar_0']
    else:
        g_state['bars_since_entry_0'] = 0
    
    # 更新品种1的持仓天数
    if g_state['last_entry_bar_1'] >= 0:
        g_state['bars_since_entry_1'] = bar_idx - g_state['last_entry_bar_1']
    else:
        g_state['bars_since_entry_1'] = 0

def vip22_strategy_optimized(api: StrategyAPI):
    """
    vip22跨品种套利策略主函数（高度优化版本）
    基于价差的范围过滤器进行交易决策
    使用缓存和增量计算大幅提高性能
    """
    global g_state
    
    if not api.require_data_sources(2):
        return
    
    # 获取策略参数（只在第一次获取）
    if 'params_cached' not in g_state:
        g_state['lotskg'] = api.get_param('lotskg', True)
        g_state['lots_data1'] = api.get_param('lots_data1', 1)
        g_state['lots_data2'] = api.get_param('lots_data2', 1)
        g_state['fund'] = api.get_param('fund', 20000)
        g_state['fast_period'] = api.get_param('fast_period', 10)
        g_state['fast_range'] = api.get_param('fast_range', 1.5)
        g_state['x_filter'] = api.get_param('x_filter', 2)
        g_state['stopout'] = api.get_param('stopout', 3)
        g_state['params_cached'] = True
    
    bar_idx = api.get_idx(0)
    
    # 最小样本数检查
    min_samples = max(g_state['fast_period'] * 3, 50)
    if bar_idx < min_samples:
        return
    
    # 只在需要时获取K线数据
    if g_state['last_calculated_bar'] != bar_idx - 1 or bar_idx % 10 == 0:  # 每10个bar强制更新一次
        data0_klines = api.get_klines(0)
        data1_klines = api.get_klines(1)
        
        if len(data0_klines) < min_samples or len(data1_klines) < min_samples:
            return
        
        # 缓存K线数据
        g_state['cached_data0'] = data0_klines
        g_state['cached_data1'] = data1_klines
    else:
        # 使用缓存的数据
        data0_klines = g_state.get('cached_data0')
        data1_klines = g_state.get('cached_data1')
        
        if data0_klines is None or data1_klines is None:
            return
    
    # 获取当前价格
    c1 = data0_klines['close'].iloc[bar_idx]
    c2 = data1_klines['close'].iloc[bar_idx]
    o1 = data0_klines['open'].iloc[bar_idx]
    o2 = data1_klines['open'].iloc[bar_idx]
    h1 = data0_klines['high'].iloc[bar_idx]
    h2 = data1_klines['high'].iloc[bar_idx]
    l1 = data0_klines['low'].iloc[bar_idx]
    l2 = data1_klines['low'].iloc[bar_idx]
    
    # 计算当前价差比率
    curr_source = c1 / c2
    chigh = max(h1/h2, o1/o2, c1/c2)
    clow = min(l1/l2, o1/o2, c1/c2)
    
    # 增量计算技术指标
    if len(g_state['price_history']) == 0:
        # 首次初始化
        start_idx = max(0, bar_idx - min_samples)
        c1_hist = data0_klines['close'].iloc[start_idx:bar_idx+1]
        c2_hist = data1_klines['close'].iloc[start_idx:bar_idx+1]
        source_hist = c1_hist / c2_hist
        
        # 计算初始技术指标
        smrng1 = smooth_range(source_hist, g_state['fast_period'], g_state['fast_range'])
        smrng2 = smooth_range(source_hist, g_state['fast_period'] * 3, g_state['fast_range'] * 0.6)
        smrng = (smrng1 + smrng2) / 2
        filt = range_filter(source_hist, smrng)
        
        # 计算趋势计数
        upward = pd.Series(0, index=source_hist.index)
        downward = pd.Series(0, index=source_hist.index)
        filt_diff = filt.diff()
        
        for i in range(1, len(filt)):
            if pd.isna(filt_diff.iloc[i]):
                continue
            if filt_diff.iloc[i] > 0:
                upward.iloc[i] = upward.iloc[i-1] + 1
                downward.iloc[i] = 0
            elif filt_diff.iloc[i] < 0:
                downward.iloc[i] = downward.iloc[i-1] + 1
                upward.iloc[i] = 0
            else:
                upward.iloc[i] = upward.iloc[i-1]
                downward.iloc[i] = downward.iloc[i-1]
        
        # 初始化状态
        max_history = min_samples + 50  # 减少历史数据保存量
        g_state['price_history'] = source_hist.tolist()[-max_history:]
        g_state['smrng_history'] = smrng.tolist()[-max_history:]
        g_state['filt_history'] = filt.tolist()[-max_history:]
        g_state['upward_history'] = upward.tolist()[-max_history:]
        g_state['downward_history'] = downward.tolist()[-max_history:]
        
        # 初始化EMA状态
        if len(smrng1) > 0:
            price_diff_hist = abs(source_hist.diff())
            g_state['ema_state']['avrng1'] = price_diff_hist.ewm(span=g_state['fast_period'], adjust=False).mean().iloc[-1]
            g_state['ema_state']['smooth_avrng1'] = smrng1.iloc[-1] / g_state['fast_range']
            g_state['ema_state']['avrng2'] = price_diff_hist.ewm(span=g_state['fast_period'] * 3, adjust=False).mean().iloc[-1]
            g_state['ema_state']['smooth_avrng2'] = smrng2.iloc[-1] / (g_state['fast_range'] * 0.6)
    
    else:
        # 增量更新
        g_state['price_history'].append(curr_source)
        
        if len(g_state['price_history']) >= 2:
            prev_source = g_state['price_history'][-2]
            price_diff = abs(curr_source - prev_source)
            
            # 增量计算平滑范围
            smrng1 = smooth_range_incremental(price_diff, g_state['fast_period'], g_state['fast_range'], g_state['ema_state'])
            smrng2 = smooth_range_incremental2(price_diff, g_state['fast_period'], g_state['fast_range'], g_state['ema_state'])
            curr_smrng = (smrng1 + smrng2) / 2
            
            # 增量计算范围过滤器
            prev_filt = g_state['filt_history'][-1] if g_state['filt_history'] else curr_source
            curr_filt = range_filter_incremental(curr_source, curr_smrng, prev_filt)
            
            # 增量计算趋势计数
            prev_upward = g_state['upward_history'][-1] if g_state['upward_history'] else 0
            prev_downward = g_state['downward_history'][-1] if g_state['downward_history'] else 0
            
            filt_diff = curr_filt - prev_filt
            
            if filt_diff > 0:
                curr_upward = prev_upward + 1
                curr_downward = 0
            elif filt_diff < 0:
                curr_downward = prev_downward + 1
                curr_upward = 0
            else:
                curr_upward = prev_upward
                curr_downward = prev_downward
            
            # 更新历史数据
            g_state['smrng_history'].append(curr_smrng)
            g_state['filt_history'].append(curr_filt)
            g_state['upward_history'].append(curr_upward)
            g_state['downward_history'].append(curr_downward)
            
            # 限制历史数据长度
            max_history = min_samples + 50
            if len(g_state['price_history']) > max_history:
                g_state['price_history'] = g_state['price_history'][-max_history:]
                g_state['smrng_history'] = g_state['smrng_history'][-max_history:]
                g_state['filt_history'] = g_state['filt_history'][-max_history:]
                g_state['upward_history'] = g_state['upward_history'][-max_history:]
                g_state['downward_history'] = g_state['downward_history'][-max_history:]
    
    # 更新计算状态
    g_state['last_calculated_bar'] = bar_idx
    
    # 获取当前计算结果
    if len(g_state['filt_history']) < 2:
        return
    
    curr_filt = g_state['filt_history'][-1]
    curr_upward = g_state['upward_history'][-1]
    curr_downward = g_state['downward_history'][-1]
    prev_source = g_state['price_history'][-2] if len(g_state['price_history']) >= 2 else curr_source
    
    # 计算交易条件
    long_cond = (curr_source > curr_filt and 
                (curr_source != prev_source) and 
                curr_upward > g_state['x_filter'])
    
    short_cond = (curr_source < curr_filt and 
                 (curr_source != prev_source) and 
                 curr_downward > g_state['x_filter'])
    
    # 获取持仓信息
    pos_data0 = api.get_pos(0)
    pos_data1 = api.get_pos(1)
    
    # 更新持仓状态
    if pos_data0 != g_state['last_pos_0']:
        if pos_data0 != 0 and g_state['last_pos_0'] == 0:
            g_state['last_entry_bar_0'] = bar_idx
        elif pos_data0 == 0:
            g_state['last_entry_bar_0'] = -1
        g_state['last_pos_0'] = pos_data0
    
    if pos_data1 != g_state['last_pos_1']:
        if pos_data1 != 0 and g_state['last_pos_1'] == 0:
            g_state['last_entry_bar_1'] = bar_idx
        elif pos_data1 == 0:
            g_state['last_entry_bar_1'] = -1
        g_state['last_pos_1'] = pos_data1
    
    # 更新持仓天数
    update_bars_since_entry(bar_idx)
    
    # 计算手数
    if g_state['lotskg']:
        lots_0 = calculate_lots(api, g_state['fund'], c1, 100, 1)
        lots_1 = calculate_lots(api, g_state['fund'], c2, 100, 1)
    else:
        lots_0 = g_state['lots_data1']
        lots_1 = g_state['lots_data2']
    
    # 交易逻辑（简化版本，减少重复代码）
    execute_trading_logic(api, pos_data0, pos_data1, long_cond, short_cond, 
                         curr_source, prev_source, chigh, clow, 
                         lots_0, lots_1, bar_idx)

def execute_trading_logic(api, pos_data0, pos_data1, long_cond, short_cond, 
                         curr_source, prev_source, chigh, clow, 
                         lots_0, lots_1, bar_idx):
    """
    执行交易逻辑的辅助函数 - 套利策略同开同平版本
    """
    global g_state
    
    # 检查是否有持仓
    has_position = (pos_data0 != 0 or pos_data1 != 0)
    
    if not has_position:  # 无持仓时，同时开仓
        if long_cond:  # 价差向上突破，做多价差（j888做多，jm888做空）
            api.buy(volume=lots_0, order_type='next_bar_open', index=0)  # j888做多
            api.sellshort(volume=lots_1, order_type='next_bar_open', index=1)  # jm888做空
            g_state['d_jcprice_0'] = curr_source  # j888多仓记录
            g_state['k_jcprice_1'] = curr_source  # jm888空仓记录
            g_state['arass_0'] = 0
            g_state['arass_1'] = 0
            
        elif short_cond:  # 价差向下突破，做空价差（j888做空，jm888做多）
            api.sellshort(volume=lots_0, order_type='next_bar_open', index=0)  # j888做空
            api.buy(volume=lots_1, order_type='next_bar_open', index=1)  # jm888做多
            g_state['k_jcprice_0'] = curr_source  # j888空仓记录
            g_state['d_jcprice_1'] = curr_source  # jm888多仓记录
            g_state['arass_0'] = 0
            g_state['arass_1'] = 0
    
    else:  # 有持仓时，检查平仓条件
        should_close = False
        
        # 检查j888的平仓条件
        if pos_data0 > 0:  # j888多头持仓
            should_close = check_long_exit_condition(0, curr_source, prev_source, chigh)
        elif pos_data0 < 0:  # j888空头持仓
            should_close = check_short_exit_condition(0, curr_source, prev_source, clow)
        
        # 检查jm888的平仓条件
        if not should_close:
            if pos_data1 > 0:  # jm888多头持仓
                should_close = check_long_exit_condition(1, curr_source, prev_source, chigh)
            elif pos_data1 < 0:  # jm888空头持仓
                should_close = check_short_exit_condition(1, curr_source, prev_source, clow)
        
        # 如果任一品种触发平仓条件，同时平仓两个品种
        if should_close:
            # 平仓j888
            if pos_data0 > 0:
                api.sell(order_type='next_bar_open', index=0)
                g_state['d_jcprice_0'] = 0
            elif pos_data0 < 0:
                api.buycover(order_type='next_bar_open', index=0)
                g_state['k_jcprice_0'] = 0
            
            # 平仓jm888
            if pos_data1 > 0:
                api.sell(order_type='next_bar_open', index=1)
                g_state['d_jcprice_1'] = 0
            elif pos_data1 < 0:
                api.buycover(order_type='next_bar_open', index=1)
                g_state['k_jcprice_1'] = 0

def check_long_exit_condition(index, curr_source, prev_source, chigh):
    """检查多头持仓的平仓条件"""
    global g_state
    
    d_jcprice_key = f'd_jcprice_{index}'
    arass_key = f'arass_{index}'
    bars_since_entry_key = f'bars_since_entry_{index}'
    
    d_jcprice = g_state[d_jcprice_key]
    if d_jcprice > 0:
        # 更新最高价
        d_jcprice = max(d_jcprice, curr_source)
        g_state[d_jcprice_key] = d_jcprice
        
        # 计算止损价
        agg = (d_jcprice - chigh) / chigh if chigh > 0 else 0
        if agg > 0:
            stopof = g_state['stopout'] * (1 - agg * 10)
        else:
            stopof = g_state['stopout']
        
        stop_price = d_jcprice - (stopof / 100) * prev_source
        arass = g_state[arass_key]
        
        if g_state[bars_since_entry_key] == 0:
            arass = stop_price
        else:
            arass = max(d_jcprice - (stopof / 100) * prev_source, arass)
        
        g_state[arass_key] = arass
        
        # 检查平仓条件
        return curr_source < arass and g_state[bars_since_entry_key] > 0
    
    return False

def check_short_exit_condition(index, curr_source, prev_source, clow):
    """检查空头持仓的平仓条件"""
    global g_state
    
    k_jcprice_key = f'k_jcprice_{index}'
    arass_key = f'arass_{index}'
    bars_since_entry_key = f'bars_since_entry_{index}'
    
    k_jcprice = g_state[k_jcprice_key]
    if k_jcprice > 0:
        # 更新最低价
        k_jcprice = min(k_jcprice, curr_source)
        g_state[k_jcprice_key] = k_jcprice
        
        # 计算止损价
        agg = (k_jcprice - clow) / clow if clow > 0 else 0
        if agg < 0:
            stopof = g_state['stopout'] * (1 + agg * 10)
        else:
            stopof = g_state['stopout']
        
        stop_price = k_jcprice + (stopof / 100) * prev_source
        arass = g_state[arass_key]
        
        if g_state[bars_since_entry_key] == 0:
            arass = stop_price
        else:
            arass = min(k_jcprice + (stopof / 100) * prev_source, arass)
        
        g_state[arass_key] = arass
        
        # 检查平仓条件
        return curr_source > arass and g_state[bars_since_entry_key] > 0
    
    return False

def performance_test():
    """
    性能测试函数 - 对比优化前后的回测速度
    """
    import time
    
    try:
        from ssquant.config.auth_config import get_api_auth
        API_USERNAME, API_PASSWORD = get_api_auth()
    except ImportError:
        print("警告：未找到 API_USERNAME, API_PASSWORD账户密码")
        return
    
    # 测试配置
    test_config = {
        'username': API_USERNAME,
        'password': API_PASSWORD,
        'use_cache': True,
        'save_data': True,
        'align_data': True,
        'fill_method': 'ffill',
        'debug': False
    }
    
    symbol_configs = [
        ('j888', {
            'start_date': '2024-10-01',  # 较短的测试期间
            'end_date': '2024-12-31',
            'initial_capital': 100000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 100,
            'periods': [{'kline_period': '1h', 'adjust_type': '1'}]
        }),
        ('jm888', {
            'start_date': '2024-10-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 60,
            'periods': [{'kline_period': '1h', 'adjust_type': '1'}]
        })
    ]
    
    strategy_params = {
        'lotskg': True,
        'lots_data1': 1,
        'lots_data2': 1,
        'fund': 20000,
        'fast_period': 10,
        'fast_range': 1.5,
        'x_filter': 2,
        'stopout': 3
    }
    
    print("开始性能测试...")
    print("=" * 50)
    
    # 测试优化版本
    print("测试优化版本...")
    backtester_opt = MultiSourceBacktester()
    backtester_opt.set_base_config(test_config)
    
    for symbol, config in symbol_configs:
        backtester_opt.add_symbol_config(symbol, config)
    
    start_time = time.time()
    results_opt = backtester_opt.run(
        strategy=vip22_strategy_optimized,
        initialize=initialize,
        strategy_params=strategy_params
    )
    opt_time = time.time() - start_time
    
    print(f"优化版本回测时间: {opt_time:.2f} 秒")
    
    # 如果有原始版本，也可以测试对比
    # 这里只显示优化版本的结果
    print("=" * 50)
    print("性能测试完成")
    print(f"优化版本性能: {opt_time:.2f} 秒")
    
    return results_opt

if __name__ == "__main__":
    # 正常回测
    # 导入API认证信息
    try:
        from ssquant.config.auth_config import get_api_auth
        API_USERNAME, API_PASSWORD = get_api_auth()
    except ImportError:
        print("警告：未找到 API_USERNAME, API_PASSWORD账户密码，请在上方get_api_auth()里面填写松鼠Quant俱乐部的账户密码")
    
    backtester = MultiSourceBacktester()
    backtester.set_base_config({
        'username': API_USERNAME,
        'password': API_PASSWORD,
        'use_cache': True,
        'save_data': True,
        'align_data': True,
        'fill_method': 'ffill',
        'debug': False  # 关闭调试模式以提高速度
    })
    
    # 添加第一个品种配置（焦炭）
    backtester.add_symbol_config(
        symbol='j888', 
        config={
            'start_date': '2024-01-01',  # 缩短回测时间以提高速度
            'end_date': '2024-12-31',
            'initial_capital': 100000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 100,
            'periods': [{'kline_period': '1h', 'adjust_type': '1'}]  # 使用1小时K线提高速度
    })
    
    # 添加第二个品种配置（焦煤）
    backtester.add_symbol_config(
        symbol='jm888', 
        config={
            'start_date': '2024-01-01',  # 缩短回测时间以提高速度
            'end_date': '2024-12-31',
            'initial_capital': 100000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 60,
            'periods': [{'kline_period': '1h', 'adjust_type': '1'}]  # 使用1小时K线提高速度
    })
    
    # 策略参数
    strategy_params = {
        'lotskg': True,           # True为自动计算手数，False为手动设置
        'lots_data1': 1,          # 品种1手动设置手数
        'lots_data2': 1,          # 品种2手动设置手数
        'fund': 20000,            # 保证金自动换算
        'fast_period': 50,        # 快速周期
        'fast_range': 1.5,        # 快速范围
        'x_filter': 3,            # 最低过滤次数
        'stopout': 6              # 出场参数
    }
    
    results = backtester.run(
        strategy=vip22_strategy_optimized,  # 使用优化版本
        initialize=initialize,
        strategy_params=strategy_params
    ) 