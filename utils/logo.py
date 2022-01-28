import shutil

def print_community_info():

    print_logo()

    print ('')

    print ('You are using Machine Video Editor comunity fork')

    print_discord_info()

    print_commercial_contact()

    print_dontation_link()

    print_recruitment_info()

    print('')

def print_discord_info():
    print ('Join us in discord, invite link https://discord.gg/KBQB8x9K5c')

def print_commercial_contact():
    print ('Commercial inquires send to ognjen@syntheticfactory.com (shared with the group)')

def print_logo():

    logo = []                                                       
    logo.append('                                           #####           ')
    logo.append('                                   .##############*        ')
    logo.append('                                     ###############(###   ')
    logo.append('                   **           ##*#########/(############ ') 
    logo.append('                  /&&&&%*      ######### ##   (# #######   ') 
    logo.append('                  /&&&&&&&&&/, (####### #       #*######   ') 
    logo.append('                  *&&&&&&&&&&&&&#/#####.#(      # #########') 
    logo.append('                  *&&&&&&&&&&&&&%%%%%((## /#### ########## ') 
    logo.append('           ###    *&%%%%%%%%&&&&%%%%%%%%%#(###########     ') 
    logo.append('       ########(##(%%%%%%%%&&&&&&&&&&&%%%%%%%#(######.     ') 
    logo.append('       /##########(%%%%%%%%%%%&&%%%%&&&%%%%%&&&&%%(#*      ') 
    logo.append('        ##########(%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&#.        ') 
    logo.append('  ##. ###########(/&&&&%%%%%%%%%%%%%%%&&&&&&&/             ') 
    logo.append(' ############# (##/&&&&%%&%%%%%%%%%&&&&&%*                 ') 
    logo.append('.############ #   *&&&&&&%%&%%%%%%%%(,                     ') 
    logo.append(' *########## ##   *&&&&&&&%&%%%%(#####(                    ') 
    logo.append('     ######## #   *&&&&&&%%#(###########                   ') 
    logo.append('     ######### ###*&&&&#( #############,                   ') 
    logo.append('  ################  .  ###########                         ')
    logo.append('   #############################(                          ') 
    logo.append('     ###  *######################                          ') 
    logo.append('             ####################.                         ') 
    logo.append('            *#######*      ###                             ')

    cols = shutil.get_terminal_size().columns
    for l_str in logo:
        print(l_str.center(0 if cols < 60 else cols), end="")
    #print(logo_str.center(100))

def print_dontation_link():
    print ('Support us via donation https://www.paypal.me/ognjenjaric or patreon https://www.patreon.com/machineeditor')

def print_recruitment_info():
    print ('If you would like to contribute we are looking for more members, apply via discord!')