import sys
import locale

system_locale = locale.getdefaultlocale()[0]
# system_locale may be nil
system_language = system_locale[0:2] if system_locale is not None else "en"
if system_language not in ['en','ru','zn']:
    system_language = 'en'

windows_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'simsun_01'
}

darwin_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'Apple LiSung Light'
}

linux_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'cour'
}

def get_default_ttf_font_name():
    platform = sys.platform
    if platform[0:3] == 'win': return windows_font_name_map.get(system_language, 'cour')
    elif platform == 'darwin': return darwin_font_name_map.get(system_language, 'cour')
    else: return linux_font_name_map.get(system_language, 'cour')

SID_HOT_KEY = 1

if system_language == 'en':
    StringsDB = {'S_HOT_KEY' : 'hot key'}
elif system_language == 'ru':
    StringsDB = {'S_HOT_KEY' : 'горячая клавиша'}    
elif system_language == 'zn':
    StringsDB = {'S_HOT_KEY' : '热键'}   
    