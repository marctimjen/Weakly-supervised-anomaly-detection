import sharepy
s = sharepy.connect("connecthkuhk-my.sharepoint.com")

# file_url = "https://connecthkuhk-my.sharepoint.com/personal/cyxcarol_connect_hku_hk/_layouts/15/onedrive.aspx?isAscending=true&ga=1&id=%2Fpersonal%2Fcyxcarol%5Fconnect%5Fhku%5Fhk%2FDocuments%2FUCF%2DCrime%2FUCF%5FTrain%5Ften%5Fi3d&sortField=LinkFilename"
# r = s.get(file_url)




# from office365.runtime.auth.user_credential import UserCredential
# from office365.sharepoint.client_context import ClientContext
#
#
# sharepoint_url = "https://aarhusuniversitet.sharepoint.com"
#
# # Initialize the client credentials
# user_credentials = UserCredential
#
# # create client context object
# ctx = ClientContext(sharepoint_url).with_credentials(user_credentials)
#

#
# # file_url is the sharepoint url from which you need the list of files
# list_source = ctx.web.get_folder_by_server_relative_url(file_url)
# files = list_source.files
# ctx.load(files)
# ctx.execute_query()
#
# print(files)
