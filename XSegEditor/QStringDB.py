from localization import system_language


class QStringDB():
    
    @staticmethod
    def initialize():
        lang = system_language
        
        if lang not in ['en','ru','zn']:
            lang = 'en'
        
        QStringDB.btn_poly_color_red_tip = {    'en' : 'Poly color scheme red',
                                                'ru' : 'Красная цветовая схема полигонов',
                                                'zn' : '多边形配色方案红色',
                                           }[lang]
                                           
        QStringDB.btn_poly_color_green_tip = {  'en' : 'Poly color scheme green',
                                                'ru' : 'Зелёная цветовая схема полигонов',
                                                'zn' : '多边形配色方案绿色',
                                           }[lang]
                                           
        QStringDB.btn_poly_color_blue_tip = {   'en' : 'Poly color scheme blue',
                                                'ru' : 'Синяя цветовая схема полигонов',
                                                'zn' : '多边形配色方案蓝色',
                                           }[lang]
                                           
        QStringDB.btn_view_baked_mask_tip = {   'en' : 'View baked mask',
                                                'ru' : 'Посмотреть запечёную маску',
                                                'zn' : '查看遮罩通道',
                                           }[lang]
                                           
        QStringDB.btn_poly_type_include_tip = { 'en' : 'Poly include mode',
                                                'ru' : 'Режим полигонов - включение',
                                                'zn' : '多边形包含模式',
                                           }[lang]
                                           
        QStringDB.btn_poly_type_exclude_tip = { 'en' : 'Poly exclude mode',
                                                'ru' : 'Режим полигонов - исключение',
                                                'zn' : '多边形排除方式',
                                           }[lang]        
                                           
        QStringDB.btn_undo_pt_tip = {   'en' : 'Undo point',
                                        'ru' : 'Отменить точку',
                                        'zn' : '撤消点',
                                    }[lang]      
                                     
        QStringDB.btn_redo_pt_tip = {   'en' : 'Redo point',
                                        'ru' : 'Повторить точку',
                                        'zn' : '重做点',
                                     }[lang]      
                                      
        QStringDB.btn_delete_poly_tip = {   'en' : 'Delete poly',
                                            'ru' : 'Удалить полигон',
                                            'zn' : '删除多边形',
                                           }[lang]     
                                              
        QStringDB.btn_pt_edit_mode_tip = {  'en' : 'Edit point mode ( HOLD CTRL )',
                                            'ru' : 'Режим правки точек',
                                            'zn' : '编辑点模式 ( 按住CTRL )',
                                           }[lang]    
                                           
        QStringDB.btn_prev_image_tip = {    'en' : 'Save and Prev image\nHold SHIFT : accelerate\nHold CTRL : skip non masked\n',
                                            'ru' : 'Сохранить и предыдущее изображение\nУдерживать SHIFT : ускорить\nУдерживать CTRL : пропустить неразмеченные\n',
                                            'zn' : '保存和上一张图片\n按住SHIFT : 加快\n按住CTRL : 跳过未标记的\n',
                                           }[lang]   
        QStringDB.btn_next_image_tip = {    'en' : 'Save and Next image\nHold SHIFT : accelerate\nHold CTRL : skip non masked\n',
                                            'ru' : 'Сохранить и следующее изображение\nУдерживать SHIFT : ускорить\nУдерживать CTRL : пропустить неразмеченные\n',
                                            'zn' : '保存并下一张图片\n按住SHIFT : 加快\n按住CTRL : 跳过未标记的\n',
                                           }[lang]  
                                           
                                           