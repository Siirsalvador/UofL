import discord
from discord.ext import commands


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')
    client = commands.Bot(command_prefix=" / ")


    @client.event
    async def on_ready():
        print("Bot is ready")
