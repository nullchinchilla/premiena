# premiena: a _reversible_ sound change applicator

**premiena** is a _sound change applicator_ written in Rust, useful for historical linguistics and conlanging.

It is similar in spirit to projects like zompist's [SCA2](http://www.zompist.com/sca2.html), but there's a very distinct feature: it can run sound changes _in reverse_.

This is by representing each sound change as a nondeterministic finite state automaton (NFST), according to the algorithms presented in Ronald Kaplan and Martin Kay's 1994 paper _Regular Models of Phonological Rule Systems_.

## Usage

### Forward derivation

`prmn` a command-line program that has one argument: a sound change file specified as a YAML file:

```
prmn --help
```

```
Usage: prmn <input> [-r]

Applies a premiena-style sound change file.

Positional Arguments:
  input             path to the sound change YAML file

Options:
  -r, --reverse     whether to apply in reverse
  --help            display usage information
```

It then takes in words on standard input, separated by newlines. These words are passed through the sound changes and produce the outputs.

Using [demo_jp.yaml](demo_jp.yaml) (a sound change file that roughly maps Old Japanese to Modern Japanese) as an example:

```
echo "teputepu" | prmn demo_jp.yaml
```

```
chōchō
```

Running `prmn demo_jp.yaml` will also work as an interactive prompt that will read one Old Japanese word and give you the Modern Japanese descendant after pressing Enter.

### Reverse derivation

Passing in `-r` will run `prmn` in _reverse_ mode:

```
echo "chōchō" | prmn -r demo_jp.yaml
```

```
teuteu tewuteu teutepu teutewu teputeu tewutepu tewutewu teputepu teputewu
```

For every line, the sound changes are run in reverse, and _all_ the possible ancestors (up to the 50 shortest) are listed.

**Note**: The `source` field in the YAML specifies a regex that matches phonotactically legal words in the parent language. Otherwise, many unreasonable words would be generated. For example, without the `source` field, reversing `chōchō` through `demo_jp.yaml` would lead to this less useful list:

```
teuteu tyōteu tyauteu tyouteu teputeu teuchō teuchou teuchau teyesutepu teutewu teutyō teutyou teutyau tewuteu chōteu chauteu chouteu tyōchō tyōchou tyōchau tyōtepu tyōtewu tyōtyō tyōtyou tyōtyau tyaputeu tyauchō tyauchou tyauchau tyautepu tyautewu tyautyō tyautyou tyautyau tyawuteu tyoputeu tyouchō tyouchou tyouchau tyoutepu tyoutewu tyoutyō tyoutyou tyoutyau tyowuteu tepuchō tepuchou tepuchau teputepu teputewu
```
