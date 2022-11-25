def duration(seconds: int) -> str:
    descriptions = []
    hours = seconds // 3600
    if hours >= 24:
        days = hours // 24
        if days > 0:
            descriptions.append(f'{days} 天')
            hours %= 24
    if hours > 0:
        descriptions.append(f'{hours} 小时')
        seconds %= 3600
    minutes = seconds // 60
    if minutes > 0:
        descriptions.append(f'{minutes} 分钟')
        seconds %= 60
    descriptions.append(f'{seconds} 秒')
    return ' '.join(descriptions)
